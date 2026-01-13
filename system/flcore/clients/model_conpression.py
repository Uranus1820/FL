import copy
from typing import List, Optional
import torch
import torch.nn as nn
from flcore.trainmodel.models import FedAvgCNN

def get_layer_l2_norms(layer: nn.Module) -> torch.Tensor:
    """计算单层权重的 L2 范数（按输出通道/神经元）"""
    with torch.no_grad():
        if isinstance(layer, nn.Conv2d):
            # [out_c, in_c, k, k] -> [out_c, -1] -> 每个输出通道一个范数
            return layer.weight.view(layer.weight.size(0), -1).norm(dim=1)
        elif isinstance(layer, nn.Linear):
            # [out_features, in_features] -> 每个输出神经元一个范数
            return layer.weight.norm(dim=1)
    raise ValueError(f"Unsupported layer type: {type(layer)}")


def reorder_layer_output(layer: nn.Module,
                         indices: torch.Tensor,
                         new_out_channels: int) -> None:
    """
    按 indices 重排 + 剪枝输出通道 / 神经元：
    - Conv2d: 重排 out_channels
    - Linear: 重排 out_features
    """
    with torch.no_grad():
        indices = indices.to(layer.weight.device)

        layer.weight.data = layer.weight.data[indices][:new_out_channels]
        if layer.bias is not None:
            layer.bias.data = layer.bias.data[indices][:new_out_channels]

        if isinstance(layer, nn.Conv2d):
            layer.out_channels = new_out_channels
        elif isinstance(layer, nn.Linear):
            layer.out_features = new_out_channels


def reorder_layer_input(layer: nn.Module,
                        indices: torch.Tensor,
                        new_in_channels: int) -> None:
    """
    按 indices 重排 + 剪枝输入通道：
    - Conv2d: 重排 in_channels
    - Linear: 重排 in_features
    """
    with torch.no_grad():
        indices = indices.to(layer.weight.device)

        layer.weight.data = layer.weight.data[:, indices][:, :new_in_channels]

        if isinstance(layer, nn.Conv2d):
            layer.in_channels = new_in_channels
        elif isinstance(layer, nn.Linear):
            layer.in_features = new_in_channels




def sort_and_compress_model(
        model: nn.Module,
        alpha: float = 0.5,
        layer_idx: int = 0,
):
    """
    返回: (compressed_model, compression_info)
    compression_info 是一个字典，记录了每一层保留的输出通道索引。
    """
    assert 0.0 < alpha <= 1.0, "alpha 必须在 (0, 1]"
    compressed_model = copy.deepcopy(model)

    # 使用 named_modules 以便后续能通过名字对应
    layers = [
        (name, m) for name, m in compressed_model.named_modules()
        if isinstance(m, (nn.Conv2d, nn.Linear))
    ]
    num_layers = len(layers)
    if num_layers == 0:
        return compressed_model, {}

    # 确定 Bottom 和 Top 的分界线
    all_params = list(compressed_model.parameters())
    split_param_idx = len(all_params) - layer_idx
    param_id_to_idx = {id(p): i for i, p in enumerate(all_params)}

    first_head_idx = num_layers
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

    prev_indices = None
    prev_new_width = None
    prev_original_width = None


    compression_info = {}

    for i, (name, layer) in enumerate(layers):
        # --- 1. 调整输入 (Input) ---
        if prev_indices is not None:
            if isinstance(layer, nn.Linear) and isinstance(layers[i - 1][1], nn.Conv2d):
                # Conv -> Linear 特殊处理
                pixels_per_channel = layer.in_features // prev_original_width
                expanded_indices = []
                for idx in prev_indices.tolist():
                    start = idx * pixels_per_channel
                    end = start + pixels_per_channel
                    expanded_indices.extend(range(start, end))

                expanded_indices = torch.tensor(expanded_indices, dtype=torch.long, device=layer.weight.device)
                new_in_features = prev_new_width * pixels_per_channel
                reorder_layer_input(layer, expanded_indices, new_in_features)
            else:
                reorder_layer_input(layer, prev_indices, prev_new_width)

        # --- 2. 调整输出 (Output) ---
        if i >= first_head_idx:
            # Head 层：不剪枝输出，不记录索引（意味着保留所有）
            prev_indices = None
            continue

        l2_norms = get_layer_l2_norms(layer)
        num_channels = l2_norms.size(0)
        new_width = max(1, int(round(num_channels * alpha)))
        new_width = min(new_width, num_channels)

        sorted_indices = torch.argsort(l2_norms, descending=True)
        topk_indices = sorted_indices[:new_width]

        reorder_layer_output(layer, topk_indices, new_width)

        # [新增] 记录该层的保留索引
        compression_info[name] = topk_indices.cpu()  # 存为 CPU tensor 减小传输开销

        prev_indices = topk_indices
        prev_new_width = new_width
        prev_original_width = num_channels

    return compressed_model, compression_info


if __name__ == "__main__":


    if FedAvgCNN is not None:
        # FedAvgCNN 通常有 2 个 Conv 和 2 个 Linear
        # 参数列表通常顺序为: [conv1.w, conv1.b, conv2.w, conv2.b, fc1.w, fc1.b, fc.w, fc.b] (共8个)
        net = FedAvgCNN(in_features=1, num_classes=10, dim=1024)
        print("原始模型架构:")
        print(net)

        # 假设 FLAYER 设置 layer_idx = 2
        # 意味着保留最后 2 个参数 (fc.w, fc.b)，即最后一层不压缩
        # 其他层 (Bottom) 进行 50% 压缩
        print("\n正在压缩 (alpha=0.5, layer_idx=2)...")
        shrunk_net = sort_and_compress_model(net, alpha=0.5, layer_idx=2)
        print("压缩后模型架构:")
        print(shrunk_net)

        # 简单前向测试
        dummy_input = torch.randn(1, 1, 28, 28)
        try:
            out = shrunk_net(dummy_input)
            print("\n前向传播成功，输出形状:", out.shape)
        except Exception as e:
            print("\n前向传播失败:", e)