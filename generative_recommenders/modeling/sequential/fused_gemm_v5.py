# 循环调用
# 每个k的块只会load一次，但是每个k的块会被重复读取
# 总共B*H*(N//BLOCK_SIZE_N)个kernel，每个kernel单层循环

import torch
import triton
import triton.language as tl


@triton.jit
def silu(x):
    return x * tl.sigmoid(x)

@triton.autotune(
    configs=[
        #triton.Config({"BLOCK_SIZE_N": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_SIZE_N": 32}, num_warps=4, num_stages=3),
        # triton.Config({"BLOCK_SIZE_N": 128}, num_warps=4, num_stages=3),
    ],
    key=["N"],
)
@triton.jit
def hstu_fused_attention_kernel(
    Q_ptr, K_ptr,
    Out_ptr,
    B, H, N, D :tl.constexpr,
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_out_b, stride_out_h, stride_out_n, stride_out_d,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_b = tl.program_id(0)  # Batch 维度
    pid_h = tl.program_id(1)  # Head 维度
    pid_n = tl.program_id(2)  # N的块数


    acc = tl.zeros((BLOCK_SIZE_N, D), dtype=tl.float32)
    
    k_ptrs = tl.make_block_ptr(
                base = K_ptr + pid_b*stride_kb + pid_h*stride_kh,
                shape = (N,D),
                strides = (stride_kn, stride_kd),
                offsets = (pid_n, 0),
                block_shape = (BLOCK_SIZE_N, D),
                order = (0, 1)
    )
    # tl.static_print("k_ptrs shape", k_ptrs)
    # tl.static_print("v_ptrs shape", v_ptrs)
    k = tl.load(k_ptrs)

    #TODO : 加载K和V的块， K_i V_i

    for block_q in range(0, N, BLOCK_SIZE_N):
        q_ptrs = tl.make_block_ptr(
                base = Q_ptr + pid_b*stride_qb + pid_h*stride_qh,
                shape = (N,D),
                strides = (stride_qn, stride_qd),
                offsets = (block_q, 0),
                block_shape = (BLOCK_SIZE_N, D),
                order = (0, 1)
        )
        #tl.static_print("q_ptrs shape",q_ptrs)
        o_ptrs = tl.make_block_ptr(
                base = Out_ptr + pid_b*stride_out_b + pid_h*stride_out_h,
                shape = (N,N),
                strides = (stride_out_n, stride_out_d),
                offsets = (block_q, pid_n),
                block_shape = (BLOCK_SIZE_N, BLOCK_SIZE_N),
                order = (0, 1)
        )
        q = tl.load(q_ptrs)
        qk = tl.dot(q, k.T)
        tl.store(o_ptrs, qk)
        #pass


def triton_batched_matmul_v5(q, k):  #N为padded后的长度， 输入的q, k 形状为[B, N, H*D], v形状为[B, N, H*D]
    B,N,H,D = q.shape
    
    q = q.permute(0,2,1,3).contiguous() #[B, N, H, D] -> [B, H, N, D]
    k = k.permute(0,2,1,3).contiguous()

    # 预分配输出张量
    output = torch.empty((B,H,N,N),device=q.device, dtype=q.dtype)

    # 调用 Triton 内核
    grid = lambda meta: (B, H, triton.cdiv(N, meta["BLOCK_SIZE_N"]))  # 每个batch的每个head单独计算

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # 记录开始时间
    start_event.record()
    hstu_fused_attention_kernel[grid](
        q, k,output,
        B, H, N, D,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
          # 自动调优会覆盖此值
    )
    end_event.record()
    torch.cuda.synchronize()
    print("Triton Time V5: ", start_event.elapsed_time(end_event))
    return output