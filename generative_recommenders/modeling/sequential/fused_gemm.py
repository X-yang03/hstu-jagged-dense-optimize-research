import triton
import triton.language as tl
import torch

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_D": 32}, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_D": 32}, num_warps=8),
    ],
    key=["n", "attention_dim"],
)
@triton.jit
def batched_matmul_kernel(
    q_ptr, k_ptr, attn_ptr,
    n, attention_dim,
    stride_a_batch, stride_a_m, stride_a_k,
    stride_b_batch, stride_b_n, stride_b_k,
    stride_c_batch, stride_c_m, stride_c_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    # 当前线程块处理的批次索引和位置块
    pid_batch = tl.program_id(0)
    pid_m = tl.program_id(1)  #q的分块
    pid_n = tl.program_id(2)    #k的分块

    # 初始化累加器
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # 分块加载 q 和 k
    q_ptrs = q_ptr + pid_batch * stride_a_batch + (pid_m * BLOCK_M + tl.arange(0, BLOCK_M))[:, None] * stride_a_m + tl.arange(0, BLOCK_D)[None, :] * stride_a_k
    k_ptrs = k_ptr + pid_batch * stride_b_batch + (pid_n * BLOCK_N + tl.arange(0, BLOCK_N))[None, :] * stride_b_n + tl.arange(0, BLOCK_D)[:, None] * stride_b_k

    tl.static_print("q_ptrs", q_ptrs.shape)
    tl.static_print("k_ptrs", k_ptrs.shape)
    # 分块计算矩阵乘法
    for d in range(0, attention_dim, BLOCK_D):
        q = tl.load(q_ptrs, mask=(d + tl.arange(0, BLOCK_D))[None,:] < attention_dim, other=0.0)
        k = tl.load(k_ptrs, mask=(d + tl.arange(0, BLOCK_D))[:,None] < attention_dim, other=0.0)
        acc += tl.dot(q, k)
        q_ptrs += BLOCK_D * stride_a_k
        k_ptrs += BLOCK_D * stride_b_k

    # 写回结果
    attn_ptrs = attn_ptr + pid_batch * stride_c_batch + (pid_m * BLOCK_M + tl.arange(0, BLOCK_M))[:, None] * stride_c_m + (pid_n * BLOCK_N + tl.arange(0, BLOCK_N))[None, :] * stride_c_n
    tl.store(attn_ptrs, acc.to(tl.float16 if q_ptr.dtype == tl.float16 else tl.float32))


def triton_batched_matmul(padded_q, padded_k):
    B, n, num_heads, attention_dim = padded_q.shape
    padded_q = padded_q.permute(0, 2, 1, 3).contiguous()  #[B, n, num_heads, attention_dim] -> [B, num_heads, n, attention_dim]
    padded_k = padded_k.permute(0, 2, 1, 3).contiguous()
    q = padded_q.view(B * num_heads, n, attention_dim).contiguous()
    k = padded_k.view(B * num_heads, n, attention_dim).contiguous()
    assert q.dim() == 3 and k.dim() == 3
    Bn, M, D = q.shape
    _, N, D = k.shape
    attn = torch.empty((Bn, M, N), device=q.device, dtype=q.dtype)

    # 网格配置
    grid = lambda meta : (Bn, triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))

    # 调用内核
    batched_matmul_kernel[grid](
        q, k, attn,
        M, D,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        attn.stride(0), attn.stride(1), attn.stride(2),
        #BLOCK_M=64, BLOCK_N=64, BLOCK_D=32,
    )
    return attn