# 输入Q与K V (Sum_N, head*dqk)
# 输出 attn(Sum_N, head*d)
# x_offsets = [0, len1, len1+len2, len1+len2+len3, ..., total_len]

# 如果使用triton 3.2.0， 需要在环境/lib/python3.10/site-packages/torch/_inductor/triton_heuristics.py
# 修改 from torch._C import _cuda_getCurrentRawStream as get_cuda_stream

# 如果使用triton 2.2.0， block_ptr中shape必需是int32，offsets必需是int64
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
    Q_ptr, K_ptr, V_ptr, rab_ptr,
    Out_ptr,
    attn_mask_ptr,
    x_offsets_ptr,
    B, H, N, D :tl.constexpr,  # B: batch size, H: head, N: sequence length, D: hidden size
    stride_kh, stride_kn, stride_kd,
    stride_qh, stride_qn, stride_qd,
    stride_vh, stride_vn, stride_vd,
    stride_rab_b, stride_rab_h, stride_rab_n, stride_rab_m,
    stride_mask_n, stride_mask_m,
    stride_out_h, stride_out_n, stride_out_d,
    BLOCK_SIZE_N: tl.constexpr
    
):
    pid_h = tl.program_id(0)
    pid_b = tl.program_id(1)

    start = tl.load(x_offsets_ptr + pid_b)
    end = tl.load(x_offsets_ptr + pid_b + 1)
    len_sample = end - start

    n_blocks = tl.cdiv(len_sample, BLOCK_SIZE_N)  # 向上取整

    for block_kv in range(n_blocks):  #load  K_i V_i
        k = tl.zeros((BLOCK_SIZE_N, D), dtype=tl.float32)
        v = tl.zeros((BLOCK_SIZE_N, D), dtype=tl.float32)
        if block_kv * BLOCK_SIZE_N + BLOCK_SIZE_N <= len_sample:   # 当前block在当前sequence的长度范围内
            k_block_ptrs = tl.make_block_ptr(
                base=K_ptr + pid_h * stride_kh + start * stride_kn,  # 当前sequence的起始位置
                shape = (len_sample, D),   #在triton 2.2.0中，必须加入to(tl.int32)；在triton 3.2.0中，不需要
                strides = (stride_kn, stride_kd), 
                offsets = ((block_kv * BLOCK_SIZE_N).to(tl.int32), 0),  #在triton 2.2.0 中，offsets必须是int64; triton 3.2.0中，offsets是int32
                block_shape = (BLOCK_SIZE_N, D),
                order = (0, 1)
            )

            v_block_ptrs = tl.make_block_ptr(
                base=V_ptr + pid_h * stride_vh + start * stride_vn,
                shape = (len_sample.to(tl.int32), D),
                strides = (stride_vn, stride_vd),
                offsets = ((block_kv * BLOCK_SIZE_N).to(tl.int32), 0),
                block_shape = (BLOCK_SIZE_N, D),
                order = (0, 1)
            )

            k = tl.load(k_block_ptrs)
            v = tl.load(v_block_ptrs)

        else:  # 对末尾的尾项进行单独处理，需要拆分读取
            #tl.device_print("jagged")
            k_ptrs = K_ptr + pid_h * stride_kh + start * stride_kn +\
                        (block_kv * BLOCK_SIZE_N) * stride_kn + \
                    tl.arange(0, BLOCK_SIZE_N)[:,None] * stride_kn + \
                    tl.arange(0, D)[None, :] * stride_kd
            #手写pointers, k与v仍然是(BLOCK_N,D)的形状
            v_ptrs = V_ptr + pid_h * stride_vh + start * stride_vn +\
                        (block_kv * BLOCK_SIZE_N) * stride_vn + \
                    tl.arange(0, BLOCK_SIZE_N)[:,None] * stride_vn + \
                    tl.arange(0, D)[None, :] * stride_vd
            
            #需要加入mask，将越界读取的数据置为0
            mask = (block_kv * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))[:,None] < len_sample

            k = tl.load(k_ptrs, mask=mask, other=0)
            v = tl.load(v_ptrs, mask=mask, other=0)

        for block_q in range(block_kv, n_blocks):  #load Q_j, 要求 block_q >= block_kv
            rab_ptrs = tl.make_block_ptr(  # rab shape : (B,1,N,N)
                base = rab_ptr + pid_b * stride_rab_b,
                shape = (N, N),
                strides = (stride_rab_n, stride_rab_m),
                offsets = ((block_q * BLOCK_SIZE_N).to(tl.int32), (block_kv * BLOCK_SIZE_N).to(tl.int32)),
                block_shape = (BLOCK_SIZE_N, BLOCK_SIZE_N),
                order = (0, 1)
            )
            rab = tl.load(rab_ptrs)

            if block_q * BLOCK_SIZE_N + BLOCK_SIZE_N <= len_sample:
                q_block_ptrs = tl.make_block_ptr(
                    base=Q_ptr + pid_h * stride_qh + start * stride_qn, # 当前sequence的Q起始位置
                    shape = (len_sample.to(tl.int32), D),
                    strides = (stride_qn, stride_qd),
                    offsets = ((block_q * BLOCK_SIZE_N).to(tl.int32), 0),
                    block_shape = (BLOCK_SIZE_N, D),
                    order = (0, 1)
                )
                o_block_ptrs = tl.make_block_ptr(
                    base = Out_ptr + pid_h*stride_out_h + start*stride_out_n,
                    shape = (len_sample.to(tl.int32) , D),
                    strides = (stride_out_n, stride_out_d),
                    offsets = ((block_q * BLOCK_SIZE_N).to(tl.int32), 0), #k_i (N,D) * q_j.T (D, N) -> o_ji (N, N)
                    block_shape = (BLOCK_SIZE_N, D),
                    order = (0, 1)
                )
                q = tl.load(q_block_ptrs)
                o = tl.load(o_block_ptrs)
                qk = silu(tl.dot(q, k.T, input_precision = "ieee") + rab) / N
                
                if block_kv == block_q:  #mask的处理方式
                # 因为mask是下三角的1矩阵，当block_kv < block_q时，不用做任何处理
                # 当block_kv == block_q时，需要将qk与mask相乘
                    mask_ptrs = tl.make_block_ptr(
                        base = attn_mask_ptr,
                        shape = (N, N),
                        strides = (stride_mask_n, stride_mask_m),
                        offsets = ((block_q * BLOCK_SIZE_N).to(tl.int32), (block_kv * BLOCK_SIZE_N).to(tl.int32)),
                        block_shape = (BLOCK_SIZE_N, BLOCK_SIZE_N),
                        order = (0, 1)
                    )
                    attn_mask = tl.load(mask_ptrs)
                    qk = qk * attn_mask

                attn = tl.dot(qk, v, input_precision = "ieee")
                o += attn
                tl.store(o_block_ptrs, o)
            else:
                #tl.device_print("jagged")
                q_ptrs = Q_ptr + pid_h * stride_qh + start * stride_qn +\
                        (block_q * BLOCK_SIZE_N) * stride_qn + \
                    tl.arange(0, BLOCK_SIZE_N)[:,None] * stride_qn + \
                    tl.arange(0, D)[None, :] * stride_qd
                mask = (block_q * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))[:,None] < len_sample
                q = tl.load(q_ptrs, mask=mask, other=0)

                o_ptrs = Out_ptr + pid_h*stride_out_h + start * stride_out_n +\
                        (block_q * BLOCK_SIZE_N) * stride_out_n + \
                        tl.arange(0, BLOCK_SIZE_N)[:,None] * stride_out_n + \
                        tl.arange(0, D)[None, :] * stride_out_d
                
                #q = tl.load(q_ptrs)
                o = tl.load(o_ptrs, mask=mask, other=0)
                qk = silu(tl.dot(q, k.T, input_precision = "ieee") + rab ) / N
                if block_kv == block_q:
                    mask_ptrs = tl.make_block_ptr(
                        base = attn_mask_ptr,
                        shape = (N, N),
                        strides = (stride_mask_n, stride_mask_m),
                        offsets = ((block_q * BLOCK_SIZE_N).to(tl.int32), (block_kv * BLOCK_SIZE_N).to(tl.int32)),
                        block_shape = (BLOCK_SIZE_N, BLOCK_SIZE_N),
                        order = (0, 1)
                    )
                    attn_mask = tl.load(mask_ptrs)
                    qk = qk * attn_mask
                
                attn = tl.dot(qk, v, input_precision = "ieee")
                o += attn
                tl.store(o_ptrs, o, mask=mask)


def fused_jagged_hstu(q, k, v, rab, attn_mask, head, dim, n, x_offsets):  #n为最长序列长度
    # q k v shape: (sum_N, head*d)
    # rab shape: (B, 1, n, n)
    # attn_mask shape: (1, 1, n, n)
    sum_N, _ = q.shape
    q = q.view(sum_N, head, dim).permute(1, 0, 2).contiguous() # (head, sum_N, d)
    k = k.view(sum_N, head, dim).permute(1, 0, 2).contiguous() # (head, sum_N, d)
    v = v.view(sum_N, head, dim).permute(1, 0, 2).contiguous() # (head, sum_N, d

    B = len(x_offsets) - 1
    #output = torch.zeros(B, head, n, dim, device=q.device, dtype=q.dtype)
    output = torch.zeros_like(q)  # (head, sum_N, d)

    grid = (head, B)

    fused_jagged_hstu_kernel[grid]( 
        q, k, v, rab,
        output,
        attn_mask,
        x_offsets,
        B, head, n, dim,
        k.stride(0), k.stride(1), k.stride(2),
        q.stride(0), q.stride(1), q.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        rab.stride(0), rab.stride(1), rab.stride(2), rab.stride(3),
        attn_mask.stride(2), attn_mask.stride(3),
        output.stride(0), output.stride(1), output.stride(2)
    )
    return output
    