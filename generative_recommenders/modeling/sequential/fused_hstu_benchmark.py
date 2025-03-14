import torch
import torch.nn.functional as F
from fused_hstu_attn import hstu_fused_attention_v1
from fused_gemm import triton_batched_matmul
from fused_hstu import hstu_fused_attention
from tqdm import tqdm

B, n, num_heads, head_dim = 16, 1024, 8, 32
q = torch.randn(B, n, num_heads, head_dim, device="cuda")
k = torch.randn(B, n, num_heads, head_dim, device="cuda")
v = torch.randn(B, n, num_heads, head_dim, device="cuda")
rab = torch.randn(B, n, n, device="cuda")

# 预热阶段：执行多次以消除冷启动开销
print("Warmup Triton")
for _ in tqdm(range(10)):
    warmup = hstu_fused_attention_v1(q, k, v, rab, False)
print("Warmup Triton Done")

print('Start Benchmark')

einsum_time = []
triton_time = []

for _ in tqdm(range(100)):

    q = torch.randn(B, n, num_heads, head_dim, device="cuda")
    k = torch.randn(B, n, num_heads, head_dim, device="cuda")
    v = torch.randn(B, n, num_heads, head_dim, device="cuda")

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    # 原始计算
    qk = torch.einsum("bnhd,bmhd->bhnm", q, k)
    qk = F.silu(qk)/n
    attn = torch.einsum("bhnm,bmhd->bnhd", qk, v)

    end_event.record()
    torch.cuda.synchronize()
    einsum_time.append(start_event.elapsed_time(end_event))
    #print("origin einsum Time: {}ms ".format(start_event.elapsed_time(end_event)))


    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    fused_attn = hstu_fused_attention_v1(q, k, v, rab, False)
    end_event.record()
    torch.cuda.synchronize()
    triton_time.append(start_event.elapsed_time(end_event))
    #print("Triton Time: {}ms ".format(start_event.elapsed_time(end_event)))

print(einsum_time)
print(triton_time)

print('avg origin time:', sum(einsum_time)/100)
print('avg triton time:', sum(triton_time)/100)
acc_ratio = [einsum_time[i]/triton_time[i] for i in range(100)]
print('avg speedup:', sum(einsum_time)/sum(triton_time))
print('max speedup:', max(acc_ratio))