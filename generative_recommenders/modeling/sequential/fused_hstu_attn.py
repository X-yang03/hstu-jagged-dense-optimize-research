import torch
import triton
import triton.language as tl


@triton.jit
def silu(x):
    return x * tl.sigmoid(x)

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_N": 32}, num_warps=4,num_stages=3),
        #triton.Config({"BLOCK_SIZE_N": 128}, num_warps=8),
    ],
    key=["N"],
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
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_b = tl.program_id(0)  # Batch 维度
    pid_h = tl.program_id(1)  # Head 维度


    #acc = tl.zeros((BLOCK_SIZE_N, D), dtype=tl.float32)
    
    for block_kv in range(0, N, BLOCK_SIZE_N):
        k_ptrs = tl.make_block_ptr(
                    base = K_ptr + pid_b*stride_kb + pid_h*stride_kh,
                    shape = (N,D),
                    strides = (stride_kn, stride_kd),
                    offsets = (block_kv, 0),
                    block_shape = (BLOCK_SIZE_N, D),
                    order = (0, 1)
        )
        v_ptrs = tl.make_block_ptr(
                    base = V_ptr + pid_b*stride_vb + pid_h*stride_vh,
                    shape = (N,D),
                    strides = (stride_vn, stride_vd),
                    offsets = (block_kv, 0),
                    block_shape = (BLOCK_SIZE_N, D),
                    order = (0, 1)
        )
        # tl.static_print("k_ptrs shape", k_ptrs)
        # tl.static_print("v_ptrs shape", v_ptrs)
        k = tl.load(k_ptrs)
        v = tl.load(v_ptrs)

        #计算q和output的初始指针
        q_ptrs = tl.make_block_ptr(
                    base = Q_ptr + pid_b*stride_qb + pid_h*stride_qh,
                    shape = (N,D),
                    strides = (stride_qn, stride_qd),
                    offsets = (0, 0),
                    block_shape = (BLOCK_SIZE_N, D),
                    order = (0, 1)
            )

        o_ptrs = tl.make_block_ptr(
                    base = Out_ptr + pid_b*stride_out_b + pid_h*stride_out_h,
                    shape = (N,D),
                    strides = (stride_out_n, stride_out_d),
                    offsets = (0, 0), #k_i (N,D) * q_j.T (D, N) -> o_ji (N, N)
                    block_shape = (BLOCK_SIZE_N, D),
                    order = (0, 1)
            )
        for block_q in range(0, N, BLOCK_SIZE_N):
            #加载Q的块 Q_j
            q = tl.load(q_ptrs)
            #加载output的块  O_j
            o = tl.load(o_ptrs)
            #计算Q_j * K_i, 得到QK_ji, (BLOCK_N, BLOCK_N)
            qk = tl.dot(q, k.T)
            #TODO: SiLU(QK_ji) / N

            #QK_ji * V_i, 得到QK_ji * V_i, (BLOCK_N, D)
            attn = tl.dot(qk, v)
            #O_j += QK_ji * V_i
            o += attn
            #stroe O_j
            tl.store(o_ptrs, o)
            q_ptrs = tl.advance(q_ptrs, (BLOCK_SIZE_N,0))
            o_ptrs = tl.advance(o_ptrs, (BLOCK_SIZE_N,0))
            #pass


def hstu_fused_attention_v1(q, k, v, rab, enable_rab):  #N为padded后的长度， 输入的q, k 形状为[B, N, H*D], v形状为[B, N, H*D]
    B,N,H,D = q.shape
    
    q = q.permute(0,2,1,3).contiguous() #[B, N, H, D] -> [B, H, N, D]
    k = k.permute(0,2,1,3).contiguous()
    v = v.permute(0,2,1,3).contiguous()
    rab = rab.contiguous()

    # 预分配输出张量
    output = torch.empty_like(q)

    # 调用 Triton 内核
    grid = (B, H)  # 每个batch的每个head单独计算
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # 记录开始时间
    start_event.record()
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
    end_event.record()
    torch.cuda.synchronize()
    print("Triton Time v1: ", start_event.elapsed_time(end_event))
    
    return output