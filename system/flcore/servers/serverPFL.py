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

        self.times = times
        self.eval_gap = args.eval_gap
        # 预先记录 Conv/Linear 层及其参数键，方便子模型还原
        self.conv_linear_specs = self._build_conv_linear_specs()
        self.set_clients(args, clientPFL)
        # ---------------- [新增: 初始化静态异构配置] ----------------
        print("Generating static client resources distribution...")

        # 1. 获取分布参数
        alpha_mu = getattr(args, 'alpha_mu', 0.5)
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
                'alpha': all_alphas[i],
                'J': all_Js[i]
            }
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
            self.cluster(current_round=i)  # 传入当前轮次用于聚类动态调整
            self.aggregate()
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

        self.aggregate_params = []
        self.uploaded_ids = []

        for client in self.selected_clients:
            payload = client.upload_payload
            if payload is None:
                continue

            client_id = payload["id"]
            sub_state_dict = payload["state_dict"]
            compression_info = payload["info"]
            J = payload["J"]
            alpha = payload["alpha"]  # 获取 alpha

            self.uploaded_ids.append(client_id)

            # 恢复模型参数 (restored_model)
            full_state = self._expand_submodel_state(
                base_state=self.global_model.state_dict(),
                sub_state=sub_state_dict,
                compression_info=compression_info,
            )

            restored_model = copy.deepcopy(self.global_model)
            restored_model.load_state_dict(full_state, strict=False)

            full_params_np = self.get_parameters(restored_model)
            self.aggregate_params.append((full_params_np, alpha, J, client_id))

            # [关键] 存储格式: (参数列表, alpha, 样本数, 客户端ID)
            self.aggregate_params.append((full_params_np, alpha, J, client_id))




    def cluster(self, current_round):
        if current_round == 0:
            self.clusterer.reset_history()

        if not self.aggregate_params:
            self.cluster_num = 0
            self.cluster_result = {}
            self.cluster_assignments = {}
            return 0, {}

        head_layer_names = self._get_head_layer_names()
        if not head_layer_names:
            self.cluster_num = 0
            self.cluster_result = {}
            self.cluster_assignments = {}
            return 0, {}

        head_layer_name_set = set(head_layer_names)

        vectors = []
        client_ids = []

        for params, _, _, client_id in self.aggregate_params:
            temp_model = copy.deepcopy(self.global_model)
            self.set_parameters(temp_model, params)

            layer_vectors = []
            for name, module in temp_model.named_modules():
                if not isinstance(module, (nn.Conv2d, nn.Linear)):
                    continue
                if name not in head_layer_name_set:
                    continue
                layer_type = "conv" if isinstance(module, nn.Conv2d) else "linear"
                norms = self._layer_l2_norms_from_weight(module.weight.data, layer_type)
                layer_vectors.append(norms.detach().cpu().numpy())

            if layer_vectors:
                vector = np.concatenate(layer_vectors, axis=0)
            else:
                vector = np.array([], dtype=np.float32)

            vectors.append(vector)
            client_ids.append(client_id)

        _, assignments = self.clusterer.cluster(vectors)
        cluster_result = {}
        for cid, cluster_id in zip(client_ids, assignments):
            cluster_result.setdefault(cluster_id, []).append(cid)

        self.cluster_num = len(cluster_result)
        self.cluster_result = cluster_result
        self.cluster_assignments = {
            cid: cluster_id for cid, cluster_id in zip(client_ids, assignments)
        }
        return self.cluster_num, self.cluster_result


    def aggregate(self):
        if not self.aggregate_params or not getattr(self, "cluster_result", None):
            return

        client_payload = {}
        for params, alpha, J, client_id in self.aggregate_params:
            client_payload[client_id] = (params, alpha, J)

        cluster_models = []
        for _, client_ids in self.cluster_result.items():
            if not client_ids:
                continue

            weights = []
            for cid in client_ids:
                if cid not in client_payload:
                    continue
                _, alpha, J = client_payload[cid]
                weights.append(alpha * J)

            total_weight = sum(weights)
            base_params = client_payload[client_ids[0]][0]
            agg_params = [np.zeros_like(p) for p in base_params]

            for cid, w in zip(client_ids, weights):
                params, alpha, J = client_payload[cid]
                if total_weight > 0:
                    coeff = (alpha * J) / total_weight
                else:
                    coeff = 1.0 / len(client_ids)
                for i in range(len(agg_params)):
                    agg_params[i] += params[i] * coeff

            cluster_models.append(agg_params)

        if not cluster_models:
            return

        num_clusters = len(cluster_models)
        global_params = [np.zeros_like(p) for p in cluster_models[0]]
        for cluster_params in cluster_models:
            for i in range(len(global_params)):
                global_params[i] += cluster_params[i] / num_clusters

        self.set_parameters(self.global_model, global_params)



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
        # return stats[4]
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
            compression_info: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:

        compression_info = compression_info or {}

        # 注意：即使形状一致，只要 compression_info 非空，也可能发生了“重排但未剪枝”，仍需恢复
        need_restore = bool(compression_info)

        if not need_restore:
            # 兼容完全不做压缩/不做重排的情况
            return {k: v.clone() for k, v in sub_state.items()}

        full_state: Dict[str, torch.Tensor] = {k: torch.zeros_like(v) for k, v in base_state.items()}
        handled_keys = set()

        prev_keep_indices = None  # 上一层“保留的输出通道在原模型中的索引”（topk_indices）
        prev_original_width = None  # 上一层原始 out_channels / out_features
        prev_layer_type = None

        for spec in self.conv_linear_specs:
            name = spec["name"]
            layer_type = spec["type"]
            wkey = spec["weight_key"]
            bkey = spec["bias_key"]

            if wkey not in base_state or wkey not in sub_state:
                continue

            base_w = base_state[wkey]
            sub_w = sub_state[wkey]
            device = base_w.device

            out_full, in_full = base_w.shape[0], base_w.shape[1]
            out_sub, in_sub = sub_w.shape[0], sub_w.shape[1]

            # 1) 输出通道：直接用客户端给的 topk_indices；Head 层则保留全部
            if name in compression_info:
                out_indices = compression_info[name]
                if not isinstance(out_indices, torch.Tensor):
                    out_indices = torch.tensor(out_indices, dtype=torch.long)
                out_indices = out_indices.to(device)
                # 兼容：客户端可能发的是 full permutation（alpha=1）或 top-k（alpha<1）
                if out_indices.numel() != out_sub:
                    out_indices = out_indices[:out_sub]
            else:
                # Head 层：客户端不剪枝输出也不重排输出
                if out_sub != out_full:
                    raise ValueError(f"Head layer out mismatch at {name}: {out_sub} vs {out_full}")
                out_indices = torch.arange(out_full, device=device)

            # 2) 输入通道：沿用上一层 keep_indices 的顺序（与客户端 reorder_layer_input 对偶）
            if prev_keep_indices is None:
                full_in_indices = torch.arange(in_full, device=device)
            else:
                if layer_type == "linear" and prev_layer_type == "conv":
                    # Conv -> Linear flatten 展开（与客户端一致）
                    assert prev_original_width is not None
                    assert in_full % prev_original_width == 0
                    pixels_per_channel = in_full // prev_original_width

                    expanded = []
                    for idx in prev_keep_indices.tolist():
                        start = idx * pixels_per_channel
                        end = start + pixels_per_channel
                        expanded.extend(range(start, end))

                    full_in_indices = torch.tensor(expanded, dtype=torch.long, device=device)
                else:
                    full_in_indices = prev_keep_indices.to(device)

            in_indices = full_in_indices[:in_sub]

            # 3) 回填权重
            full_w = full_state[wkey]
            if layer_type == "conv":
                full_w[out_indices.unsqueeze(1), in_indices.unsqueeze(0), :, :] = sub_w.to(device)
            else:
                full_w[out_indices.unsqueeze(1), in_indices.unsqueeze(0)] = sub_w.to(device)
            full_state[wkey] = full_w
            handled_keys.add(wkey)

            # 4) 回填 bias
            if bkey is not None and bkey in base_state and bkey in sub_state:
                full_b = full_state[bkey]
                sub_b = sub_state[bkey].to(device)

                if name in compression_info:
                    full_b[out_indices] = sub_b
                else:
                    full_b.copy_(sub_b)

                full_state[bkey] = full_b
                handled_keys.add(bkey)

            # 5) 更新 prev_*（Head 层后需 reset，与客户端一致）
            if name in compression_info:
                prev_keep_indices = out_indices
                prev_original_width = out_full
                prev_layer_type = layer_type
            else:
                prev_keep_indices = None
                prev_original_width = None
                prev_layer_type = None

        # 6) 其他非 Conv/Linear 参数：优先用 sub_state（形状一致），否则回退 base_state
        for k, v in base_state.items():
            if k in handled_keys:
                continue
            if k in sub_state and sub_state[k].shape == v.shape:
                full_state[k] = sub_state[k].clone()
            else:
                full_state[k] = v.clone()

        return full_state
