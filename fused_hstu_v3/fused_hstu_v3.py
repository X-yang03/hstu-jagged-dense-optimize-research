
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
        triton.Config({"BLOCK_SIZE_N": 16}, num_warps=4,num_stages=4),
        # triton.Config({"BLOCK_SIZE_N": 32}, num_warps=8,num_stages=3),
        # triton.Config({"BLOCK_SIZE_N": 64}, num_warps=4,num_stages=3),
        # triton.Config({"BLOCK_SIZE_N": 64}, num_warps=8, num_stages=3),
        # triton.Config({"BLOCK_SIZE_N": 128}, num_warps=8, num_stages=3),
        # triton.Config({"BLOCK_SIZE_N": 256}, num_warps=8, num_stages=3),
        # triton.Config({"BLOCK_SIZE_N": 512}, num_warps=8, num_stages=3),
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
    stride_kn, stride_kh, stride_kd,
    stride_qn, stride_qh, stride_qd,
    stride_vn, stride_vh, stride_vd,
    stride_rab_b, stride_rab_h, stride_rab_n, stride_rab_m,
    stride_mask_n, stride_mask_m,
    stride_out_n, stride_out_h, stride_out_d,
    BLOCK_SIZE_N: tl.constexpr
    
):
    pid_h = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_q = tl.program_id(2)

    start = tl.load(x_offsets_ptr + pid_b)
    end = tl.load(x_offsets_ptr + pid_b + 1)
    len_sample = (end - start).to(tl.int32)
    block_q = (pid_q * BLOCK_SIZE_N).to(tl.int32)


    if block_q + BLOCK_SIZE_N <= len_sample:  # q块在范围内
        o = tl.zeros((BLOCK_SIZE_N, D), dtype=tl.float32)
        # q 是 （N, H, D）形状
        q_blk_ptrs = tl.make_block_ptr(
            base = Q_ptr + pid_h * stride_qh + start * stride_qn,
            shape = ((len_sample * H), D),
            strides = (stride_qn, stride_qd),  #实际上 stride_qn = stride_qh * H
            offsets = (block_q , 0),
            block_shape = (BLOCK_SIZE_N, D),
            order = (0, 1)
        )
        
        q = tl.load(q_blk_ptrs)

        for block_kv in range(0, block_q + BLOCK_SIZE_N, BLOCK_SIZE_N):  #load  K_j V_j
            #当q在范围内时，由于kv<=q，所以kv也在范围内
            k_blk_ptrs = tl.make_block_ptr(
            base = K_ptr + pid_h * stride_kh + start * stride_kn,
            shape = ((len_sample * H), D),
            strides = (stride_kn, stride_kd),
            offsets = (block_kv , 0),
            block_shape = (BLOCK_SIZE_N, D),
            order = (0, 1)
        )
            v_blk_ptrs = tl.make_block_ptr(
            base = V_ptr + pid_h * stride_vh + start * stride_vn,
            shape = ((len_sample * H), D),
            strides = (stride_vn, stride_vd),
            offsets = (block_kv , 0),
            block_shape = (BLOCK_SIZE_N, D),
            order = (0, 1)
        )

            k = tl.load(k_blk_ptrs)
            v = tl.load(v_blk_ptrs)

            rab_blk_ptrs = tl.make_block_ptr(  # rab shape : (B,1,N,N)
                base = rab_ptr + pid_b * stride_rab_b,
                shape = (N, N),
                strides = (stride_rab_n, stride_rab_m),
                offsets = (block_q , block_kv ),
                block_shape = (BLOCK_SIZE_N, BLOCK_SIZE_N),
                order = (0, 1)
            )
            rab = tl.load(rab_blk_ptrs)

            qk = silu(tl.dot(q, k.T, input_precision = "ieee") + rab) / N

            if block_kv == block_q:  #mask的处理方式,由于kv和q都在范围内，mask可以用blk_ptr读取
                # 因为mask是下三角的1矩阵，当block_kv < block_q时，不用做任何处理
                # 当block_kv == block_q时，需要将qk与mask相乘
                    mask_ptrs = tl.make_block_ptr(
                        base = attn_mask_ptr,
                        shape = (N, N),
                        strides = (stride_mask_n, stride_mask_m),
                        offsets = (block_q , block_kv),
                        block_shape = (BLOCK_SIZE_N, BLOCK_SIZE_N),
                        order = (0, 1)
                    )
                    attn_mask = tl.load(mask_ptrs)
                    qk = qk * attn_mask
            
            attn = tl.dot(qk, v, input_precision = "ieee")
            o += attn
            
        o_block_ptrs = tl.make_block_ptr(
                    base = Out_ptr + pid_h*stride_out_h + start*stride_out_n,
                    shape = ((len_sample * H) , D),
                    strides = (stride_out_n, stride_out_d),
                    offsets = (block_q , 0), #k_i (N,D) * q_j.T (D, N) -> o_ji (N, N)
                    block_shape = (BLOCK_SIZE_N, D),
                    order = (0, 1)
                )
        tl.store(o_block_ptrs, o)


    else:
        if block_q < len_sample: # 末尾是不规则长度

            o = tl.zeros((BLOCK_SIZE_N, D), dtype=tl.float32)

            q_ptrs = Q_ptr + start * stride_qn + block_q * stride_qn + pid_h * stride_qh +\
                    tl.arange(0, BLOCK_SIZE_N)[:,None] * stride_qn + \
                    tl.arange(0, D)[None, :] * stride_qd
            
            mask = (block_q + tl.arange(0, BLOCK_SIZE_N))[:,None] < len_sample
            q = tl.load(q_ptrs, mask=mask, other=0)

            for block_kv in range(0, block_q + BLOCK_SIZE_N, BLOCK_SIZE_N):  #load  K_j V_j
                rab_ptrs = rab_ptr + pid_b * stride_rab_b + \
                    block_q * stride_rab_n + block_kv * stride_rab_m + \
                    tl.arange(0, BLOCK_SIZE_N)[:,None] * stride_rab_n + \
                    tl.arange(0, BLOCK_SIZE_N)[None, :] * stride_rab_m
                
                if block_kv + BLOCK_SIZE_N <= len_sample:  #kv块在范围内，此时kv小于q，不用乘以mask
                    k_blk_ptrs = tl.make_block_ptr(
                    base = K_ptr + pid_h * stride_kh + start * stride_kn,
                    shape = ((len_sample * H), D),
                    strides = (stride_kn, stride_kd),
                    offsets = (block_kv , 0),
                    block_shape = (BLOCK_SIZE_N, D),
                    order = (0, 1)
                )
                    v_blk_ptrs = tl.make_block_ptr(
                    base = V_ptr + pid_h * stride_vh + start * stride_vn,
                    shape = ((len_sample * H), D),
                    strides = (stride_vn, stride_vd),
                    offsets = (block_kv , 0),
                    block_shape = (BLOCK_SIZE_N, D),
                    order = (0, 1)
                )

                    k = tl.load(k_blk_ptrs)
                    v = tl.load(v_blk_ptrs)

                    rab = tl.load(rab_ptrs, mask = mask, other=0)

                    qk = silu(tl.dot(q, k.T, input_precision = "ieee") + rab) / N

                    attn = tl.dot(qk, v, input_precision = "ieee")
                    o += attn

                else: #此时q和kv都在末尾，q == kv
                    k_ptrs = K_ptr + start * stride_kn + block_kv * stride_kn + pid_h * stride_kh +\
                    tl.arange(0, BLOCK_SIZE_N)[:,None] * stride_kn + \
                    tl.arange(0, D)[None, :] * stride_kd

                    v_ptrs = V_ptr + start * stride_vn + block_kv * stride_vn + pid_h * stride_vh +\
                            tl.arange(0, BLOCK_SIZE_N)[:,None] * stride_vn + \
                            tl.arange(0, D)[None, :] * stride_vd

                    #需要加入mask，将越界读取的数据置为0
                    mask_kv = (block_kv + tl.arange(0, BLOCK_SIZE_N))[:,None] < len_sample

                    k = tl.load(k_ptrs, mask=mask_kv, other=0)
                    v = tl.load(v_ptrs, mask=mask_kv, other=0)

                    rab = tl.load(rab_ptrs, mask= mask & mask_kv.T, other=0)
                    qk = silu(tl.dot(q, k.T, input_precision = "ieee") + rab) / N

                    # if block_kv == block_q:  #mask的处理方式
                        # 因为mask是下三角的1矩阵，当block_kv < block_q时，不用做任何处理
                        # 当block_kv == block_q时，需要将qk与mask相乘
                    
                    #注意：kv和q都在末尾不规则长度，此时mask不能用blk_ptr读取！！！
                    mask_ptrs = attn_mask_ptr + block_q * stride_mask_n + block_kv * stride_mask_m +\
                            tl.arange(0, BLOCK_SIZE_N)[:,None] * stride_mask_n +\
                            tl.arange(0, BLOCK_SIZE_N)[None,:] * stride_mask_m
                    attn_mask = tl.load(mask_ptrs, mask = mask & mask_kv.T, other=0)
                    qk = qk * attn_mask
                    
                    attn = tl.dot(qk, v, input_precision = "ieee")
                    o += attn
            
            o_ptrs = Out_ptr + start * stride_out_n + block_q * stride_out_n + pid_h * stride_out_h +\
                    tl.arange(0, BLOCK_SIZE_N)[:,None] * stride_out_n + \
                    tl.arange(0, D)[None, :] * stride_out_d
            tl.store(o_ptrs, o, mask = mask)


        return


def fused_jagged_hstu(q, k, v, rab, attn_mask, head, dim, n, x_offsets):  #n为最长序列长度


    B = len(x_offsets) - 1
    output = torch.zeros_like(q)  # (head, sum_N, d)

    grid = lambda meta:(head, B, triton.cdiv(n, meta["BLOCK_SIZE_N"]))

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
    