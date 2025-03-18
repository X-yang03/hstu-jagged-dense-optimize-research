# 输入Q与K (Sum_N, head*dqk)
# x_offsets = [0, len1, len1+len2, len1+len2+len3, ...]

import torch
import triton
import triton.language as tl

@triton.jit
def silu(x):
    return x*tl.sigmoid(x) 


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_N": 32}, num_warps=4,num_stages=3),
        #triton.Config({"BLOCK_SIZE_N": 128}, num_warps=8),
    ],
    key=["N"],
)

@triton.jit
def fused_jagged_hstu_kernel(
    Q_ptr, K_ptr, V_ptr,
    Out_ptr,
    x_offsets_ptr,
    B, H, N, D :tl.constexpr,  # B: batch size, H: head, N: sequence length, D: hidden size
    stride_kh, stride_kn, stride_kd,
    stride_qh, stride_qn, stride_qd,
    stride_vh, stride_vn, stride_vd,
    stride_out_b, stride_out_h, stride_out_n, stride_out_d,
    BLOCK_SIZE_N: tl.constexpr
    
):
    pid_h = tl.program_id(0)
    pid_b = tl.program_id(1)

    start = tl.load(x_offsets_ptr + pid_b)
    end = tl.load(x_offsets_ptr + pid_b + 1)
    len_sample = end - start

    n_blocks = tl.cdiv(len_sample, BLOCK_SIZE_N)

    for block_kv in range(n_blocks):  #load  K_i V_i
        k = tl.zeros((BLOCK_SIZE_N, D), dtype=tl.float32)
        v = tl.zeros((BLOCK_SIZE_N, D), dtype=tl.float32)
        if block_kv * BLOCK_SIZE_N + BLOCK_SIZE_N <= len_sample:   # 当前block的长度小于BLOCK_SIZE_N，直接读取
            k_block_ptrs = tl.make_block_ptr(
                base=K_ptr + pid_h * stride_kh + start * stride_kn,  # 当前sequence的起始位置
                shape = (len_sample.to(tl.int32), D),   #在triton 2.2.0中，必须加入to(tl.int32)；在triton 3.2.0中，不需要
                strides = (stride_kn, stride_kd),
                offsets = (block_kv * BLOCK_SIZE_N, 0),  #在triton 2.2.0 中，offsets必须是int64; triton 3.2.0中，offsets是int32
                block_shape = (BLOCK_SIZE_N, D),
                order = (0, 1)
            )

            v_block_ptrs = tl.make_block_ptr(
                base=V_ptr + pid_h * stride_vh + start * stride_vn,
                shape = (len_sample.to(tl.int32), D),
                strides = (stride_vn, stride_vd),
                offsets = ((block_kv * BLOCK_SIZE_N).to(tl.int64), 0),
                block_shape = (BLOCK_SIZE_N, D),
                order = (0, 1)
            )

            k = tl.load(k_block_ptrs)
            v = tl.load(v_block_ptrs)
        else:  # 当前block的长度大于BLOCK_SIZE_N，需要拆分读取
            k_ptrs = K_ptr + pid_h * stride_kh + start * stride_kn +\
                        (block_kv * BLOCK_SIZE_N) * stride_kn + \
                    tl.arange(0, BLOCK_SIZE_N)[:,None] * stride_kn + \
                    tl.arange(0, D)[None, :] * stride_kd
            #手写pointers, k与v仍然是(BLOCK_N,D)的形状
            v_ptrs = V_ptr + pid_h * stride_vh + start * stride_vn +\
                        (block_kv * BLOCK_SIZE_N) * stride_vn + \
                    tl.arange(0, BLOCK_SIZE_N)[:,None] * stride_vn + \
                    tl.arange(0, D)[None, :] * stride_vd
            
            mask = (block_kv * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))[:,None] < len_sample

            k = tl.load(k_ptrs, mask=mask, other=0)
            v = tl.load(v_ptrs, mask=mask, other=0)

        for block_q in range(n_blocks):  #load Q_j
            if block_q * BLOCK_SIZE_N + BLOCK_SIZE_N <= len_sample:
                q_block_ptrs = tl.make_block_ptr(
                    base=Q_ptr + pid_h * stride_qh + start * stride_qn, # 当前sequence的Q起始位置
                    shape = (len_sample.to(tl.int32), D),
                    strides = (stride_qn, stride_qd),
                    offsets = ((block_q * BLOCK_SIZE_N), 0),
                    block_shape = (BLOCK_SIZE_N, D),
                    order = (0, 1)
                )
                o_block_ptrs = tl.make_block_ptr(
                    base = Out_ptr + pid_b*stride_out_b + pid_h*stride_out_h,
                    shape = (N,D),
                    strides = (stride_out_n, stride_out_d),
                    offsets = ((block_q * BLOCK_SIZE_N), 0), #k_i (N,D) * q_j.T (D, N) -> o_ji (N, N)
                    block_shape = (BLOCK_SIZE_N, D),
                    order = (0, 1)
                )
                q = tl.load(q_block_ptrs)
                o = tl.load(o_block_ptrs)
                qk = silu(tl.dot(q, k.T))/N
                
                attn = tl.dot(qk, v)
                o += attn
                tl.store(o_block_ptrs, o)
            else:
                q_ptrs = Q_ptr + pid_h * stride_qh + start * stride_qn +\
                        (block_q * BLOCK_SIZE_N) * stride_qn + \
                    tl.arange(0, BLOCK_SIZE_N)[:,None] * stride_qn + \
                    tl.arange(0, D)[None, :] * stride_qd
                mask = (block_q * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))[:,None] < len_sample
                q = tl.load(q_ptrs, mask=mask, other=0)

                o_ptrs = Out_ptr + pid_b*stride_out_b + pid_h*stride_out_h +\
                        (block_q * BLOCK_SIZE_N) * stride_out_n + \
                        tl.arange(0, BLOCK_SIZE_N)[:,None] * stride_out_n + \
                        tl.arange(0, D)[None, :] * stride_out_d
                
                #q = tl.load(q_ptrs)
                o = tl.load(o_ptrs, mask=mask, other=0)
                qk = silu(tl.dot(q, k.T))/N
                attn = tl.dot(qk, v)
                o += attn
                tl.store(o_ptrs, o, mask=mask)


def fused_jagged_hstu(q, k, v, head, dim, n, x_offsets):  #n为最长序列长度
    # q k v shape: (sum_N, head*d)
    sum_N, _ = q.shape
    q = q.view(sum_N, head, dim).permute(1, 0, 2).contiguous() # (head, sum_N, d)
    k = k.view(sum_N, head, dim).permute(1, 0, 2).contiguous() # (head, sum_N, d)
    v = v.view(sum_N, head, dim).permute(1, 0, 2).contiguous() # (head, sum_N, d

    B = len(x_offsets) - 1
    output = torch.zeros(B, head, n, dim, device=q.device, dtype=q.dtype)

    grid = (head, B)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    # 记录开始时间
    start_event.record()

    fused_jagged_hstu_kernel[grid]( 
        q, k, v,
        output,
        x_offsets,
        B, head, n, dim,
        k.stride(0), k.stride(1), k.stride(2),
        q.stride(0), q.stride(1), q.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
    )
    # 记录结束时间
    end_event.record()
    torch.cuda.synchronize()
    print("Triton Time: {}ms ".format(start_event.elapsed_time(end_event)))

    return output
    