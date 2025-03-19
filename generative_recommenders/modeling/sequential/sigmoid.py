import torch
import triton
import triton.language as tl

@triton.jit
def sigmoid_kernel(x_ptr, out_ptr, N):
    pid = tl.program_id(0)
    if pid < N:
        x = tl.load(x_ptr + pid)  # 读取输入数据
        y = tl.sigmoid(x)  # 使用 Triton 计算 sigmoid
        tl.store(out_ptr + pid, y)  # 将结果写回

# 生成测试数据
N = 1024
x = torch.randn(N, device="cuda")  # 在 GPU 上生成随机数据
out_triton = torch.empty_like(x)  # 预分配 Triton 输出
out_torch = torch.sigmoid(x)  # 计算 torch.sigmoid 结果

# 运行 Triton 内核
grid = (N,)
sigmoid_kernel[grid](x, out_triton, N)

# 计算误差
error = torch.abs(out_triton - out_torch)
max_error = error.max().item()
mean_error = error.mean().item()

print(out_triton[:10])
print(out_torch[:10])
print(f"最大误差: {max_error:.6f}")
print(f"平均误差: {mean_error:.6f}")
