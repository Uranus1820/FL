import copy
from typing import List, Optional
import torch
import torch.nn as nn


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
    skip_last: int = 0,
) -> nn.Module:
    """
    通道排序 + 子模型抽取（子模型训练比例 alpha）：

    - 先按每层输出通道 L2 范数排序；
    - 只对“底部” feature 部分做输出通道剪枝；
    - 跳过最后 `skip_last` 个 conv/linear 的输出剪枝（作为 head）；
    - 不论 skip_last 取值如何，最后一层总是只调整输入，不剪输出（类别数保持不变）。

    Args:
        model: 全局模型（不会被原地修改，会先 deepcopy）
        alpha: 保留通道比例 (0, 1]，例如 0.5 表示保留前 50% 通道
        skip_last: 不做输出剪枝的“尾部层”数量（以 Conv/Linear 层为单位）
                   例如 skip_last=2 表示最后 2 个 conv/linear 视为 head，不压缩输出。
    """
    assert 0.0 < alpha <= 1.0, "alpha 必须在 (0, 1]"

    compressed_model = copy.deepcopy(model)

    # 只考虑 Conv2d / Linear
    layers: List[nn.Module] = [
        m for m in compressed_model.modules()
        if isinstance(m, (nn.Conv2d, nn.Linear))
    ]
    num_layers = len(layers)
    if num_layers == 0:
        return compressed_model

    # 至少保证最后一层不做输出剪枝
    head_output_layers = max(skip_last, 1)
    head_output_layers = min(head_output_layers, num_layers)
    first_head_idx = num_layers - head_output_layers  # i >= first_head_idx 的层不裁剪输出

    prev_indices: Optional[torch.Tensor] = None
    prev_new_width: Optional[int] = None
    prev_original_width: Optional[int] = None

    for i, layer in enumerate(layers):
        # ----------------- 1) 先根据上一层（若被压缩）调整本层输入 -----------------
        if prev_indices is not None:
            if isinstance(layer, nn.Linear) and isinstance(layers[i - 1], nn.Conv2d):
                # Conv -> Linear, 需要按通道展开后的 index 做映射
                assert prev_original_width is not None
                assert layer.in_features % prev_original_width == 0, \
                    "Linear.in_features 不能被上一层通道数整除，请检查模型结构。"

                pixels_per_channel = layer.in_features // prev_original_width

                expanded_indices = []
                for idx in prev_indices.tolist():
                    start = idx * pixels_per_channel
                    end = start + pixels_per_channel
                    expanded_indices.extend(range(start, end))

                expanded_indices = torch.tensor(
                    expanded_indices,
                    dtype=torch.long,
                    device=layer.weight.device,
                )
                new_in_features = prev_new_width * pixels_per_channel  # type: ignore[arg-type]
                reorder_layer_input(layer, expanded_indices, new_in_features)
            else:
                # Conv -> Conv 或 Linear -> Linear
                reorder_layer_input(layer, prev_indices, prev_new_width)  # type: ignore[arg-type]

        # ----------------- 2) 决定是否对该层做“输出通道剪枝” -----------------
        # i >= first_head_idx 的层视为 head，只调整输入，不剪输出
        if i >= first_head_idx:
            # head 层不做输出剪枝，也不再把 index 传下去
            prev_indices = None
            prev_new_width = None
            prev_original_width = None
            continue

        # feature 层：做 L2 排序 + 输出剪枝
        l2_norms = get_layer_l2_norms(layer)
        num_channels = l2_norms.size(0)

        new_width = max(1, int(round(num_channels * alpha)))
        new_width = min(new_width, num_channels)

        sorted_indices = torch.argsort(l2_norms, descending=True)
        topk_indices = sorted_indices[:new_width]

        reorder_layer_output(layer, topk_indices, new_width)

        # 记录给下一层当作“上一层被压缩的输出映射”
        prev_indices = topk_indices
        prev_new_width = new_width
        prev_original_width = num_channels

    return compressed_model


# if __name__ == "__main__":
#     # 仅用于本文件单独运行时的简单自测，不影响作为库被 import
#     try:
#         from trainmodel.models import FedAvgCNN  # pragma: no cover
#     except Exception:
#         FedAvgCNN = None
#
#     if FedAvgCNN is not None:
#         net = FedAvgCNN(input_dim=16 * 5 * 5, hidden_dims=[120, 84], output_dim=10)
#         print(net)
#
#         shrunk_net = sort_and_compress_model(net, alpha=0.5, skip_last=1)
#         print(shrunk_net)
#
#         dummy_input = torch.randn(1, 3, 32, 32)
#         out = shrunk_net(dummy_input)
#         print("前向传播 OK，输出形状:", out.shape)
