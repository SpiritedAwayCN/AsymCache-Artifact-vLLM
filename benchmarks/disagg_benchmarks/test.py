import torch

# 假设 rebuilt_block_tables 是一个包含多个 list[tensor] 的列表
rebuilt_block_tables = [
    [torch.randn(1), torch.randn(1)],
    [torch.randn(1), torch.randn(1)],
]

print(rebuilt_block_tables)

result = torch.tensor(rebuilt_block_tables)
print(result)