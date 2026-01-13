import copy
import numpy as np
import torch
import time
import openpyxl as op
import random
from typing import Dict, List, Optional, Tuple
from functools import reduce
from collections import OrderedDict
from flcore.clients.clientPFL import *
from utils.data_utils import read_client_data
from threading import Thread
import torch.nn as nn
from collections import defaultdict
# [新增] 引入聚类模块
from flcore.servers.Cluster import EnhancedDynamicGMMClusterer


class FLAYER(object):
    def __init__(self, args, times):
        self.args = args
        self.device = args.device
        self.dataset = args.dataset
        self.global_rounds = args.global_rounds
        self.global_model = copy.deepcopy(args.model)
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.random_join_ratio = args.random_join_ratio
        self.join_clients = int(self.num_clients * self.join_ratio)
        self.layer_idx = args.layer_idx  # 记录 layer_idx 以区分 Top/Bottom
        self.batch_size = args.batch_size
        self.clients = []
        self.selected_clients = []

        self.uploaded_ids = []
        self.aggregate_params = []  # 存储格式: [(params, alpha, J), ...]

        self.rs_test_acc = []
        self.rs_train_loss = []
        self.state_keys = list(self.global_model.state_dict().keys())
        self.times = times
        self.eval_gap = args.eval_gap
        # 预先记录 Conv/Linear 层及其参数键，方便子模型还原
        self.conv_linear_specs = self._build_conv_linear_specs()
        self.set_clients(args, clientPFL)

        self.uploaded_packages: List[Dict] = []
        self.cluster_models: Dict[int, Dict[str, torch.Tensor]] = {}
        # ---------------- [新增: 初始化静态异构配置] ----------------
        print("Generating static client resources distribution...")

        # 1. 获取分布参数
        alpha_mu = getattr(args, 'alpha_mu', 0.7)
        alpha_std = getattr(args, 'alpha_std', 0.1)
        J_mu = getattr(args, 'J_mu', 200)
        J_std = getattr(args, 'J_std', 20)

        # 2. 为所有客户端 (self.num_clients) 生成固定的 alpha 和 J
        # 注意：这里是为“所有”客户端生成，不仅仅是选中的
        all_alphas = np.random.normal(loc=alpha_mu, scale=alpha_std, size=self.num_clients)
        all_alphas = np.clip(all_alphas, 0.1, 1.0)  # 截断

        all_Js = np.random.normal(loc=J_mu, scale=J_std, size=self.num_clients)
        all_Js = np.maximum(all_Js, 10).astype(int)  # 截断

        # 3. 将配置绑定到 client_id，存入字典方便后续查找
        # 假设 self.clients 里的顺序和 0到N 的索引是一一对应的
        self.client_profiles = {}
        for i, client in enumerate(self.clients):
            # client.id 通常是整数索引
            self.client_profiles[client.id] = {
                'alpha': round(all_alphas[i], 1),
                'J': all_Js[i]
            }
            #print(self.client_profiles[client.id]['alpha'])


        self.wb = op.Workbook()
        self.ws = self.wb['Sheet']

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")


        self.Budget = []
        self.clusterer = EnhancedDynamicGMMClusterer(args)

    def train(self):
        accs = [0.0] * self.num_clients

        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models(accs)
            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                accs, test_acc = self.evaluate(nonprint=None)


            # for client in self.selected_clients:
            #     client.train()
            for client in self.selected_clients:
                # 根据 client.id 获取该客户端固定的 alpha 和 J
                profile = self.client_profiles[client.id]
                # 调用客户端训练，传入它固定的属性
                client.train(alpha=profile['alpha'], J=profile['J'])
            # 接收并聚类
            self.receive_models()
            self.cluster_aggregate(current_round=i)  # 传入当前轮次用于聚类动态调整

            self.Budget.append(time.time() - s_t)

            print('-' * 50, self.Budget[-1])

        print("\nBest global accuracy.")
        print(max(self.rs_test_acc))
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))

    def set_clients(self, args, clientObj):
        for i in range(self.num_clients):
            train_data = read_client_data(self.dataset, i, is_train=True)
            test_data = read_client_data(self.dataset, i, is_train=False)
            client = clientObj(args,
                               id=i,
                               train_samples=len(train_data),
                               test_samples=len(test_data))
            self.clients.append(client)

    def select_clients(self):
        if self.random_join_ratio:
            join_clients = np.random.choice(range(self.join_clients, self.num_clients + 1), 1, replace=False)[0]
        else:
            join_clients = self.join_clients
        selected_clients = list(np.random.choice(self.clients, join_clients, replace=False))
        return selected_clients

    def send_models(self, accs):
        assert (len(self.clients) > 0)
        for client, acc in zip(self.clients, accs):
            client.local_initialization(self.global_model, acc)

    def receive_models(self):
        """
        接收并恢复客户端模型，同时存储 alpha 和 J。
        """
        assert len(self.selected_clients) > 0
        self.uploaded_ids = []
        self.uploaded_packages = []
        for client in self.selected_clients:
            pkg = client.upload_payload
            if pkg is None:
                continue

            client_id = pkg["id"]
            sub_state_dict = pkg["state_dict"]

            self.uploaded_ids.append(client_id)

            # 恢复模型参数 (restored_model)
            #full_state = self._expand_submodel_state( self.global_model.state_dict(), sub_state_dict)

            pkg["state_dict"] = sub_state_dict
            self.uploaded_packages.append(pkg)



    def cluster_aggregate(self, current_round):
        if not self.uploaded_packages:
            return

        # 1) 抽取各客户端分类头特征，运行动态 GMM 聚类，得到簇中心与分配结果
        classifier_vectors = [self._extract_classifier_vector(pkg["state_dict"]) for pkg in self.uploaded_packages]
        centers, assignments = self.clusterer._dynamic_gmm(classifier_vectors)

        self.cluster_models = {}
        # 2) 将属于同一簇的上传包索引聚在一起，方便后续逐簇聚合
        cluster_clients: Dict[int, List[int]] = defaultdict(list)
        for idx, cluster_id in enumerate(assignments):
            cluster_clients[cluster_id].append(idx)
        # 3) 针对每个簇：
        print(cluster_clients.items())
        for cluster_id, pkg_indices in cluster_clients.items():
            pkgs = [self.uploaded_packages[i] for i in pkg_indices]
            # Compute alpha*J weights for clients in this cluster



            # alpha_j_values = [float(pkg.get("alpha", 0.0)) * float(pkg.get("J", 0.0)) for pkg in pkgs]
            alpha_j_values = [ float(pkg.get("J", 0.0)) for pkg in pkgs]
            total_alpha_j = sum(alpha_j_values)
            if total_alpha_j <= 0:
                # Fallback to uniform weights if metadata is missing or invalid
                weights = [1.0 / len(pkgs)] * len(pkgs)
            else:
                weights = [val / total_alpha_j for val in alpha_j_values]

            # Weighted aggregation within the cluster
            cluster_state = {k: torch.zeros_like(v) for k, v in pkgs[0]["state_dict"].items()}
            for pkg, w in zip(pkgs, weights):
                for k, v in pkg["state_dict"].items():
                    cluster_state[k] += v * w

            self.cluster_models[cluster_id] = cluster_state

        # 4) Aggregate cluster models with equal weights 1/K to form global model
        if not self.cluster_models:
            return
        num_clusters = len(self.cluster_models)
        cluster_weight = 1.0 / num_clusters
        first_cluster_state = next(iter(self.cluster_models.values()))
        global_state = {k: torch.zeros_like(v) for k, v in first_cluster_state.items()}
        for cluster_state in self.cluster_models.values():
            for k, v in cluster_state.items():
                global_state[k] += v * cluster_weight
        self.global_model.load_state_dict(global_state)



    def _extract_classifier_vector(self, state_dict: Dict[str, torch.Tensor]) -> np.ndarray:
        """将分类头参数展平为向量，用于度量客户端之间的相似度。"""
        parts = []
        for key in self._classifier_keys():
            parts.append(state_dict[key].detach().cpu().view(-1))
        return torch.cat(parts).numpy()
    def _classifier_keys(self) -> List[str]:
        """返回被视作“分类头”的参数键名（仅后 layer_idx 层）。"""
        return self.state_keys[-self.args.layer_idx:]

    def _feature_keys(self) -> List[str]:
        """返回被视作“底部特征提取器”的参数键名。"""
        return self.state_keys[:-self.args.layer_idx]
    def _flatten_state(self, state_dict: Dict[str, torch.Tensor]) -> np.ndarray:
        """将完整模型拉平成单个向量，方便计算欧氏距离。"""
        flat = []
        for key in self.state_keys:
            flat.append(state_dict[key].detach().cpu().view(-1))
        return torch.cat(flat).numpy()



    def set_parameters(self, model, parameters):
        for new_param, old_param in zip(parameters, model.parameters()):
            old_param.data = torch.tensor(new_param, dtype=torch.float).to(self.device)

    def get_parameters(self, model):
        return [val.data.cpu().numpy() for val in model.parameters()]

        # Helper functions
        #   - 子模型还原相关工具
        # ------------------------------------------------------------------
    def _build_conv_linear_specs(self) -> List[Dict]:
        """
        预先从全局模型中抽取所有 Conv2d / Linear 层及其参数键顺序，
            用于在服务器端将各客户端上传的“子模型”还原成完整模型。
        """
        specs: List[Dict] = []
        for name, module in self.global_model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                weight_key = f"{name}.weight" if name else "weight"
                bias_key = f"{name}.bias" if getattr(module, "bias", None) is not None else None
                layer_type = "conv" if isinstance(module, nn.Conv2d) else "linear"
                specs.append({
                    "name": name,
                    "weight_key": weight_key,
                    "bias_key": bias_key,
                    "type": layer_type,
                })
        return specs
    def _layer_l2_norms_from_weight(self, weight: torch.Tensor, layer_type: str) -> torch.Tensor:
        """
        按输出通道 / 神经元计算 L2 范数（与客户端通道排序逻辑保持一致）。
        """
        if layer_type == "conv":
            return weight.view(weight.size(0), -1).norm(dim=1)
        elif layer_type == "linear":
            return weight.norm(dim=1)
        else:
            raise ValueError(f"Unsupported layer_type: {layer_type}")

    def _get_head_layer_names(self) -> List[str]:
        layers = [
            (name, m) for name, m in self.global_model.named_modules()
            if isinstance(m, (nn.Conv2d, nn.Linear))
        ]
        if not layers:
            return []

        all_params = list(self.global_model.parameters())
        split_param_idx = len(all_params) - self.layer_idx
        param_id_to_idx = {id(p): i for i, p in enumerate(all_params)}

        first_head_idx = len(layers)
        for i, (name, layer) in enumerate(layers):
            layer_params = [p for p in layer.parameters(recurse=False)]
            is_layer_bottom = True
            for p in layer_params:
                if param_id_to_idx.get(id(p), -1) >= split_param_idx:
                    is_layer_bottom = False
                    break
            if not is_layer_bottom:
                first_head_idx = i
                break

        return [name for name, _ in layers[first_head_idx:]]

    def _expand_submodel_state(
            self,
            base_state: Dict[str, torch.Tensor],
            sub_state: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        将客户端上传的“子模型”参数（通道数被压缩、顺序被重排）还原为
        与 base_state 同形状的完整模型参数，并对缺失通道做 zero-padding。

        还原规则与客户端的 sort_and_compress_model 中的通道排序 / 剪枝逻辑对偶：
        - 对每一层 Conv/Linear：
          * 根据 base_state 计算通道 L2 范数并排序，得到输出通道重排顺序；
          * 根据上一层的排序结果，推导本层输入通道的重排顺序（含 Conv->Linear flatten）；
          * 将子模型中保留下来的通道权重映射回对应原通道位置，其余位置补 0。

        若 sub_state 与 base_state 在所有 Conv/Linear 层的形状完全一致，
        认为该客户端未做结构压缩，直接返回 sub_state 的拷贝。
        """
        # 先检查是否真的做了结构压缩：如果所有 Conv/Linear 权重形状都一致，则不处理
        is_compressed = False
        for spec in self.conv_linear_specs:
            wkey = spec["weight_key"]
            if wkey in base_state and wkey in sub_state:
                if base_state[wkey].shape != sub_state[wkey].shape:
                    is_compressed = True
                    break
        if not is_compressed:
            # 兼容当前“只做 mask，不改 shape”的实现
            return {k: v.clone() for k, v in sub_state.items()}

        # 用 base_state 的形状初始化一个全零的完整模型
        full_state: Dict[str, torch.Tensor] = {k: torch.zeros_like(v) for k, v in base_state.items()}
        handled_keys = set()

        prev_indices: Optional[torch.Tensor] = None  # 上一层输出通道在原模型中的排序（完整长度）
        prev_original_width: Optional[int] = None  # 上一层原始输出通道数
        prev_layer_type: Optional[str] = None

        num_layers = len(self.conv_linear_specs)

        for i, spec in enumerate(self.conv_linear_specs):
            layer_type = spec["type"]
            wkey = spec["weight_key"]
            bkey = spec["bias_key"]

            if wkey not in base_state or wkey not in sub_state:
                continue

            base_w = base_state[wkey]
            sub_w = sub_state[wkey]
            device = base_w.device

            if layer_type == "conv":
                out_full, in_full = base_w.shape[0], base_w.shape[1]
                out_sub, in_sub = sub_w.shape[0], sub_w.shape[1]
            elif layer_type == "linear":
                out_full, in_full = base_w.shape[0], base_w.shape[1]
                out_sub, in_sub = sub_w.shape[0], sub_w.shape[1]
            else:
                continue

            # -------- 1) 输出通道映射：子模型 -> 原模型 --------
            if i == num_layers - 1:
                # 最后一层客户端不会做输出通道剪枝，只调整输入
                out_indices = torch.arange(out_full, device=device)
                if out_sub != out_full:
                    raise ValueError(f"Final layer out features mismatch: {out_sub} vs {out_full}")
            else:
                l2_norms = self._layer_l2_norms_from_weight(base_w, layer_type)
                sorted_indices = torch.argsort(l2_norms, descending=True)
                # 子模型有多少通道，就取前多少个原通道
                out_indices = sorted_indices[:out_sub]

            # -------- 2) 输入通道映射：子模型 -> 原模型 --------
            if prev_indices is None:
                full_in_indices = torch.arange(in_full, device=device)
            else:
                if layer_type == "linear" and prev_layer_type == "conv":
                    # Conv -> Linear, 需要展开成按通道连续的索引（与 sort_and_compress_model 一致）
                    assert prev_original_width is not None
                    assert in_full % prev_original_width == 0, (
                        f"Linear.in_features {in_full} is not divisible by prev_original_width {prev_original_width}"
                    )
                    pixels_per_channel = in_full // prev_original_width
                    expanded = []
                    for idx in prev_indices.tolist():
                        start = idx * pixels_per_channel
                        end = start + pixels_per_channel
                        expanded.extend(range(start, end))
                    full_in_indices = torch.tensor(expanded, dtype=torch.long, device=device)
                else:
                    # Conv->Conv 或 Linear->Linear：沿用上一层的排序
                    full_in_indices = prev_indices.to(device)

            # 只取前 in_sub 个输入通道，和客户端剪枝后的一致
            in_indices = full_in_indices[:in_sub]

            # -------- 3) 将子模型权重写回完整模型位置 --------
            full_w = full_state[wkey]
            if layer_type == "conv":
                # [子 out, 子 in, kH, kW] -> [原 out_indices, 原 in_indices, kH, kW]
                full_w[out_indices.unsqueeze(1), in_indices.unsqueeze(0), :, :] = sub_w.to(device)
            else:
                # [子 out, 子 in] -> [原 out_indices, 原 in_indices]
                full_w[out_indices.unsqueeze(1), in_indices.unsqueeze(0)] = sub_w.to(device)
            full_state[wkey] = full_w
            handled_keys.add(wkey)

            # -------- 4) 处理 bias --------
            if bkey is not None and bkey in base_state and bkey in sub_state:
                base_b = base_state[bkey]
                sub_b = sub_state[bkey]
                full_b = full_state[bkey]

                if i == num_layers - 1:
                    # 最后一层：输出维度不变，bias 一一对应拷贝
                    if base_b.shape != sub_b.shape:
                        raise ValueError(f"Final layer bias shape mismatch: {sub_b.shape} vs {base_b.shape}")
                    full_b.copy_(sub_b.to(device))
                else:
                    # 只给保留下来的输出通道赋值，其余保持 0
                    full_b[out_indices] = sub_b.to(device)
                full_state[bkey] = full_b
                handled_keys.add(bkey)

            # -------- 5) 更新给下一层用的 prev_indices / prev_original_width --------
            if i != num_layers - 1:
                l2_norms = self._layer_l2_norms_from_weight(base_w, layer_type)
                prev_indices = torch.argsort(l2_norms, descending=True).to(device)
                prev_original_width = out_full
                prev_layer_type = layer_type
            else:
                prev_indices = None
                prev_original_width = None
                prev_layer_type = None

        # -------- 6) 其他非 Conv/Linear 参数（如 BN 等），直接从子模型拷贝 --------
        for key, base_tensor in base_state.items():
            if key in handled_keys:
                continue
            if key in sub_state:
                full_state[key] = sub_state[key].to(base_tensor.device)
            else:
                full_state[key] = base_tensor.clone()

        return full_state

    def test_metrics(self):
        num_samples = []
        tot_correct = []
        tot_auc = []
        accs = []
        for c in self.clients:
            ct, ns, auc = c.test_metrics()
            # print(f'Client {c.id}: Acc: {ct*1.0/ns}, AUC: {auc}')
            tot_correct.append(ct * 1.0)
            tot_auc.append(auc * ns)
            num_samples.append(ns)
            accs.append(ct * 1.0 / ns)
        ids = [c.id for c in self.clients]

        return ids, num_samples, tot_correct, tot_auc, accs

    def train_metrics(self):
        num_samples = []
        losses = []
        for c in self.clients:
            cl, ns = c.train_metrics()
            # print(f'Client {c.id}: Train loss: {cl*1.0/ns}')
            num_samples.append(ns)
            losses.append(cl * 1.0)

        ids = [c.id for c in self.clients]

        return ids, num_samples, losses


    def evaluate(self, acc=None, loss=None, nonprint=None):
        stats = self.test_metrics()
        stats_train = self.train_metrics()

        test_acc = sum(stats[2]) * 1.0 / sum(stats[1])
        test_auc = sum(stats[3]) * 1.0 / sum(stats[1])
        train_loss = sum(stats_train[2]) * 1.0 / sum(stats_train[1])
        accs = [a / n for a, n in zip(stats[2], stats[1])]
        aucs = [a / n for a, n in zip(stats[3], stats[1])]
        losses = [a / n for a, n in zip(stats_train[2], stats_train[1])]

        # 每轮保存这个准确率.
        # data = []
        # data.append(test_acc)
        # data += accs
        # self.ws.append(data)
        # filename = "cifar100_cnn_flayer.xlsx"
        # self.wb.save(filename)

        if nonprint == None:
            if acc == None:
                self.rs_test_acc.append(test_acc)
            else:
                acc.append(test_acc)

            if loss == None:
                self.rs_train_loss.append(train_loss)
            else:
                loss.append(train_loss)

            print("Averaged Train Loss: {:.4f}".format(train_loss))
            print("Averaged Test Accurancy: {:.4f}".format(test_acc))
            #print("Averaged Test AUC: {:.4f}".format(test_auc))
            #print("Std Test Accurancy: {:.4f}".format(np.std(accs)))
            #print("Std Test AUC: {:.4f}".format(np.std(aucs)))
        # else:
        #     return accs
        return accs, test_acc
        #return stats[4]
