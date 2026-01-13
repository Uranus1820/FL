import torch
import torch.nn as nn
import copy
import numpy as np

# 假设你的文件结构如下，请根据实际情况调整 import
# 这里的 sort_and_compress_model 是你修改后的支持 layer_idx 的版本
from flcore.clients.model_conpression import sort_and_compress_model


# 这里的 restore_model_from_state_dict 是你在 serverPFL.py 中新加的方法
# 为了方便测试，我直接把 restore 函数复制在这里，或者你可以从 serverPFL 导入
# from serverPFL import FLAYER (如果把方法写在类里不太好导，建议单独定义或复制下方的 restore 函数)

# ==========================================
# 1. 定义测试用的简单模型
# ==========================================
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Conv1: 1 -> 4
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, padding=1)
        # Conv2: 4 -> 8
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, padding=1)
        # Flatten: 8 * 4 * 4 (假设输入 4x4) = 128
        # Linear1: 128 -> 10
        self.fc = nn.Linear(8 * 4 * 4, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# ==========================================
# 2. 复制服务器端的恢复逻辑 (如果你没法直接导入)
# ==========================================
def restore_model_from_state_dict(original_model, sub_state_dict, compression_info):
    """
    (这里粘贴上一轮提供的 restore_model_from_state_dict 函数代码)
    为了脚本可运行，我简略写出核心逻辑，请确保使用你最新实现的版本。
    """
    restored_model = copy.deepcopy(original_model)
    orig_layers = dict(restored_model.named_modules())
    target_layer_names = [name for name, m in restored_model.named_modules() if isinstance(m, (nn.Conv2d, nn.Linear))]

    prev_indices = None
    prev_original_width = None

    for i, name in enumerate(target_layer_names):
        orig_layer = orig_layers[name]

        # --- 1. Determine Input Indices ---
        if i == 0:
            col_indices = torch.arange(orig_layer.weight.size(1))
        else:
            if prev_indices is None:
                col_indices = torch.arange(orig_layer.weight.size(1))
            else:
                prev_layer_name = target_layer_names[i - 1]
                prev_layer_type = type(orig_layers[prev_layer_name])
                if isinstance(orig_layer, nn.Linear) and issubclass(prev_layer_type, nn.Conv2d):
                    pixels_per_channel = orig_layer.in_features // prev_original_width
                    expanded_indices = []
                    for idx in prev_indices.tolist():
                        start = idx * pixels_per_channel
                        end = start + pixels_per_channel
                        expanded_indices.extend(range(start, end))
                    col_indices = torch.tensor(expanded_indices, dtype=torch.long)
                else:
                    col_indices = prev_indices

        # --- 2. Determine Output Indices ---
        if name in compression_info:
            row_indices = compression_info[name]
            prev_indices = row_indices
            prev_original_width = orig_layer.weight.size(0)
        else:
            row_indices = torch.arange(orig_layer.weight.size(0))
            prev_indices = None
            prev_original_width = None

        # --- 3. Fill Weights ---
        weight_key = name + ".weight"
        bias_key = name + ".bias"

        if weight_key in sub_state_dict:
            sub_weight = sub_state_dict[weight_key]

            # Move indices to device
            if isinstance(row_indices, torch.Tensor): row_indices = row_indices.to(sub_weight.device)
            if isinstance(col_indices, torch.Tensor): col_indices = col_indices.to(sub_weight.device)

            with torch.no_grad():
                new_weight = torch.zeros_like(orig_layer.weight)  # 初始化为 0

                # 核心验证逻辑：填回去
                if isinstance(orig_layer, nn.Linear):
                    temp = new_weight[row_indices]
                    temp[:, col_indices] = sub_weight
                    new_weight[row_indices] = temp
                else:
                    temp = new_weight[row_indices]
                    temp[:, col_indices, :, :] = sub_weight
                    new_weight[row_indices] = temp

                orig_layer.weight.data = new_weight

                if orig_layer.bias is not None and bias_key in sub_state_dict:
                    sub_bias = sub_state_dict[bias_key]
                    new_bias = torch.zeros_like(orig_layer.bias)
                    new_bias[row_indices] = sub_bias
                    orig_layer.bias.data = new_bias

    return restored_model


# ==========================================
# 3. 执行验证
# ==========================================
def verify():
    # 1. 初始化原始模型
    print(">>> 1. 初始化原始模型...")
    original_model = SimpleCNN()
    # 为了方便验证，我们手动把权重设为特定的值，而不是随机值
    # 例如：全 1，或者按顺序赋值，这样更容易看出来是否对齐
    with torch.no_grad():
        original_model.conv1.weight.fill_(1.0)
        original_model.conv2.weight.fill_(2.0)
        original_model.fc.weight.fill_(3.0)

    # 2. 压缩模型 (Alpha=0.5, layer_idx=2)
    # layer_idx=2 意味着保留 fc 层 (weight+bias)
    print("\n>>> 2. 执行压缩 (Alpha=0.5, layer_idx=2)...")
    model_small, info = sort_and_compress_model(original_model, alpha=0.5, layer_idx=2)

    print(f"   压缩信息 (保留的通道索引): {info}")
    print(f"   小模型 Conv1 形状: {model_small.conv1.weight.shape}")
    print(f"   小模型 Conv2 形状: {model_small.conv2.weight.shape}")
    print(f"   小模型 FC 形状: {model_small.fc.weight.shape}")

    # 3. 恢复模型
    print("\n>>> 3. 执行恢复...")
    restored_model = restore_model_from_state_dict(original_model, model_small.state_dict(), info)

    # 4. 验证逻辑
    print("\n>>> 4. 开始验证数据一致性...")

    # --- 验证 Conv1 ---
    # 原始形状 [4, 1, 3, 3], 压缩后 [2, 1, 3, 3]
    # 恢复后应该形状是 [4, 1, 3, 3]
    # 其中 info['conv1'] 对应的行应该有值，其余行为 0
    layer_name = 'conv1'
    indices = info[layer_name]

    w_restored = restored_model.conv1.weight
    w_small = model_small.conv1.weight

    # 检查1: 形状是否恢复
    assert w_restored.shape == original_model.conv1.weight.shape, "Conv1 形状恢复失败"

    # 检查2: 被选中的通道值是否一致
    # restored 的 [indices] 行 应该等于 small 的所有行
    w_restored_selected = w_restored[indices]
    if torch.allclose(w_restored_selected, w_small):
        print(f"   [Pass] {layer_name} 被选中的通道数据一致。")
    else:
        print(f"   [Fail] {layer_name} 数据不匹配！")

    # 检查3: 未被选中的通道是否为 0
    all_indices = torch.arange(w_restored.shape[0])
    # 找出不在 indices 中的索引
    mask = torch.ones(w_restored.shape[0], dtype=torch.bool)
    mask[indices] = False
    zero_indices = all_indices[mask]

    if torch.sum(torch.abs(w_restored[zero_indices])) == 0:
        print(f"   [Pass] {layer_name} 未选中的通道已正确置零。")
    else:
        print(f"   [Fail] {layer_name} 未选中的通道不为 0！")

    # --- 验证 FC (Head 层) ---
    # 这里的输入维度应该也是稀疏的，因为上一层 Conv2 被压缩了
    # 但 FC 输出维度应该完整
    layer_name = 'fc'
    # FC 不在 info 中，因为它是 Head 层

    w_restored = restored_model.fc.weight
    w_small = model_small.fc.weight

    # 检查1: 形状完全恢复
    assert w_restored.shape == original_model.fc.weight.shape, "FC 形状恢复失败"

    # 检查2: 权重是否一致？
    # 注意：FC 的 *输入* 维度对应上一层 Conv2 的输出。
    # Conv2 被压缩了，所以 FC 的输入列中，只有对应 Conv2 保留通道的部分有值，其他列应为 0。
    # 而小模型 model_small.fc 是已经把输入维度缩小了的。
    # 所以我们不能直接比较 w_restored == w_small，需要按列索引比较。

    prev_indices = info['conv2']  # 上一层保留的索引
    # 计算 Conv2 -> Linear 展开后的列索引
    pixels_per_channel = original_model.fc.in_features // original_model.conv2.out_channels
    expanded_col_indices = []
    for idx in prev_indices:
        start = idx * pixels_per_channel
        end = start + pixels_per_channel
        expanded_col_indices.extend(range(start, end))

    w_restored_selected_cols = w_restored[:, expanded_col_indices]

    # 小模型的 FC 权重应该是完整的（因为是 Head 层），但其 shape[1] 变小了
    # 它们的值应该对应 restored 中被选中的那些列
    if torch.allclose(w_restored_selected_cols, w_small):
        print(f"   [Pass] {layer_name} (Head层) 对应的输入列数据一致。")
    else:
        print(f"   [Fail] {layer_name} 数据不匹配！")
        print("Restored shape subset:", w_restored_selected_cols.shape)
        print("Small shape:", w_small.shape)

    print("\n>>> ✅ 验证完成！如果上方没有 Fail，则逻辑正确。")


if __name__ == '__main__':
    verify()