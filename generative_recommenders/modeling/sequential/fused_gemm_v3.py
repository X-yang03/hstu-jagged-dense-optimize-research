import triton
import triton.language as tl
import torch

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 16}, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}, num_warps=8),
    ],
    key=["n", "attention_dim"],
)
@triton.jit
def batched_matmul_kernel(
    q_ptr, k_ptr, attn_ptr,
    n, attention_dim : tl.constexpr,
    stride_a_batch, stride_a_m, stride_a_k,
    stride_b_batch, stride_b_n, stride_b_k,
    stride_c_batch, stride_c_m, stride_c_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    # 当前线程块处理的批次索引和位置块
    pid_batch = tl.program_id(0)
    pid_m = tl.program_id(1)  #q的分块
    pid_n = tl.program_id(2)    #k的分块

    # 初始化累加器
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # 分块加载 q 和 k,形状分别为(M,D)和(N,D)
    q_ptrs = q_ptr + pid_batch * stride_a_batch + (pid_m * BLOCK_M + tl.arange(0, BLOCK_M))[:, None] * stride_a_m + tl.arange(0, attention_dim)[None, :] * stride_a_k
    k_ptrs = k_ptr + pid_batch * stride_b_batch + (pid_n * BLOCK_N + tl.arange(0, BLOCK_N))[None, :] * stride_b_n + tl.arange(0, attention_dim)[:, None] * stride_b_k
    # 分块计算矩阵乘法
    q = tl.load(q_ptrs)
    k = tl.load(k_ptrs)
    acc += tl.dot(q, k)

    # 写回结果
    attn_ptrs = attn_ptr + pid_batch * stride_c_batch + (pid_m * BLOCK_M + tl.arange(0, BLOCK_M))[:, None] * stride_c_m + (pid_n * BLOCK_N + tl.arange(0, BLOCK_N))[None, :] * stride_c_n
    tl.store(attn_ptrs, acc.to(tl.float16 if q_ptr.dtype == tl.float16 else tl.float32))


def triton_batched_matmul_v3(padded_q, padded_k):
    B, n, num_heads, attention_dim = padded_q.shape
    padded_q = padded_q.permute(0, 2, 1, 3)  #[B, n, num_heads, attention_dim] -> [B, num_heads, n, attention_dim]
    padded_k = padded_k.permute(0, 2, 1, 3)
    q = padded_q.view(B * num_heads, n, attention_dim).contiguous()
    k = padded_k.view(B * num_heads, n, attention_dim).contiguous()
    assert q.dim() == 3 and k.dim() == 3
    Bn, M, D = q.shape  # M和N其实是长度n
    _, N, D = k.shape
    attn = torch.empty((Bn, M, N), device=q.device, dtype=q.dtype)

    # 网格配置
    grid = lambda meta : (Bn, triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # 记录开始时间
    start_event.record()
    # 调用内核
    batched_matmul_kernel[grid](
        q, k, attn,
        M, D,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        attn.stride(0), attn.stride(1), attn.stride(2),
        #BLOCK_M=64, BLOCK_N=64, BLOCK_D=32,
    )
    # 记录结束时间
    end_event.record()
    torch.cuda.synchronize()
    print("Triton Time V3: ", start_event.elapsed_time(end_event))
    return attn