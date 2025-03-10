import torch
import triton
import triton.language as tl


@triton.jit
def silu(x):
    return x * tl.sigmoid(x)

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64}, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128}, num_warps=8),
    ],
    key=["N", "D"],
)
@triton.jit
def hstu_fused_attention_kernel(
    Q_ptr, K_ptr, V_ptr, rab_ptr,
    Out_ptr,
    B, H, N, D :tl.constexpr,
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_rab_b, stride_rab_n, stride_rab_m,
    stride_out_b, stride_out_h, stride_out_n, stride_out_d,
    enable_rab,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_b = tl.program_id(0)  # Batch 维度
    pid_h = tl.program_id(1)  # Head 维度
    pid_m = tl.program_id(2)  # 输出位置 (N)


    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # 计算 Q 的块指针
    offs_q_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)  #计算offs_q_m范围的token的attention
    offs_q_d = tl.arange(0, D)
    q_ptrs = Q_ptr + pid_b * stride_qb + pid_h * stride_qh + offs_q_m[:, None] * stride_qn + offs_q_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=(offs_q_m[:, None] < N) & (offs_q_d[None, :] < D), other=0.0) #q形状(B, H, N, D), 分块后q[BLOCK_M, D]
    #tl.static_print("q shape", q.shape)
    # 遍历 K 的块
    for block_n in range(0, tl.cdiv(N, BLOCK_SIZE_N)):
        offs_n = block_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        
        k_ptrs = K_ptr + pid_b * stride_kb + pid_h * stride_kh + offs_n[:, None] * stride_kn + offs_q_d[None, :] * stride_kd
        k = tl.load(k_ptrs, mask=(offs_n[:, None] < N) & (offs_q_d[None, :] < D), other=0.0)  #k也是分块后的k[BLOCK_N, D]，所以qk = q@k.T
        tl.static_print("k shape", k.shape)

        v_ptrs = V_ptr + pid_b * stride_vb + pid_h * stride_vh + offs_n[:, None] * stride_vn + offs_q_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=(offs_n[:, None] < N) & (offs_q_d[None, :] < D), other=0.0)
        #tl.static_print("v shape", v.shape) #v形状为[BLOCK_N, D]

        # 计算 Q•K 的部分和
        qk = tl.dot(q, k.T) #qk形状为[BLOCK_M, BLOCK_N]
        
        # 加载 rab 块并相加
        if enable_rab:
            rab_ptrs = rab_ptr + pid_b * stride_rab_b + offs_q_m[:, None] * stride_rab_n + offs_n[None, :] * stride_rab_m
            rab = tl.load(rab_ptrs, mask=(offs_q_m[:, None] < N) & (offs_n[None, :] < N), other=0.0)
            qk += rab

        # 应用 SiLU 激活函数并归一化
        qk = silu(qk) / N
        attn = tl.dot(qk, v)  #qk形状为[BLOCK_M, BLOCK_N], v形状为[BLOCK_N, D], attn形状为[BLOCK_M, D]
        tl.static_assert(attn.shape == acc.shape)

        # 乘累加到 acc
        acc += tl.make_contiguous(attn)

    # 存储输出
    offs_out_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_out_d = tl.arange(0, D)
    out_ptrs = Out_ptr + pid_b * stride_out_b + pid_h * stride_out_h + offs_out_m[:, None] * stride_out_n + offs_out_d[None, :] * stride_out_d
    tl.store(out_ptrs, acc, mask=(offs_out_m[:, None] < N) & (offs_out_d[None, :] < D))


def hstu_fused_attention(q, k, v, rab, enable_rab):  #N为padded后的长度， 输入的q, k 形状为[B, N, H*D], v形状为[B, N, H*D]
    B,N,H,D = q.shape
    
    q = q.permute(0,2,1,3).contiguous() #[B, N, H, D] -> [B, H, N, D]
    k = k.permute(0,2,1,3).contiguous()
    v = v.permute(0,2,1,3).contiguous()
    rab = rab.contiguous()

    # 预分配输出张量
    output = torch.empty_like(q)

    # 调用 Triton 内核
    grid = lambda meta : (B, H, triton.cdiv(N, meta["BLOCK_SIZE_M"]))  # 调整 BLOCK_SIZE_M
    hstu_fused_attention_kernel[grid](
        q, k, v, rab, output,
        B, H, N, D,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        rab.stride(0), rab.stride(1), rab.stride(2),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        enable_rab=enable_rab,
          # 自动调优会覆盖此值
    )
    return output