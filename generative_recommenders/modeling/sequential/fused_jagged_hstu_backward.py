import torch
import triton
import triton.language as tl
@triton.jit
def silu(x):
    return x*tl.sigmoid(x) 

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_N": 32}, num_warps=4,num_stages=4),
    ],
    key=["N"],
)


@triton.jit
def fused_backward_kernel(
    Q_ptr, K_ptr, V_ptr, rab_ptr,
    dQ_ptr, dK_ptr, dV_ptr, 
    Out_ptr,
    dOut_ptr,
    attn_mask_ptr,
    x_offsets_ptr,
    B, H, N, D :tl.constexpr,
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
    len_sample = (end - start).to(tl.int32)

    for block_kv in range(0, len_sample, BLOCK_SIZE_N):  #load  K_i V_i
        k = tl.zeros((BLOCK_SIZE_N, D), dtype=tl.float32)
        v = tl.zeros((BLOCK_SIZE_N, D), dtype=tl.float32)
        k_block_ptrs = tl.make_block_ptr(
                base=K_ptr + pid_h * stride_kh + start * stride_kn,  # 当前sequence的起始位置
                shape = (len_sample, D),   #在triton 2.2.0中，必须加入to(tl.int32)；在triton 3.2.0中，不需要
                strides = (stride_kn, stride_kd), 
                offsets = (block_kv, 0),  #在triton 2.2.0 中，offsets必须是int64; triton 3.2.0中，offsets是int32
                block_shape = (BLOCK_SIZE_N, D),
                order = (0, 1)
            )

        v_block_ptrs = tl.make_block_ptr(
            base=V_ptr + pid_h * stride_vh + start * stride_vn,
            shape = (len_sample.to(tl.int32), D),
            strides = (stride_vn, stride_vd),
            offsets = (block_kv, 0),
            block_shape = (BLOCK_SIZE_N, D),
            order = (0, 1)
        )

        dk_block_ptrs = tl.make_block_ptr(
            base=dK_ptr + pid_h * stride_kh + start * stride_kn,
            shape = (len_sample.to(tl.int32), D),
            strides = (stride_kn, stride_kd),
            offsets = (block_kv, 0),
            block_shape = (BLOCK_SIZE_N, D),
            order = (0, 1)
        )

        dv_block_ptrs = tl.make_block_ptr(
            base=dV_ptr + pid_h * stride_vh + start * stride_vn,
            shape = (len_sample.to(tl.int32), D),
            strides = (stride_vn, stride_vd),
            offsets = (block_kv, 0),
            block_shape = (BLOCK_SIZE_N, D),
            order = (0, 1)
        )

        k = tl.load(k_block_ptrs)
        v = tl.load(v_block_ptrs)

        d_k = tl.zeros((BLOCK_SIZE_N, D), dtype=tl.float32)
        d_v = tl.zeros((BLOCK_SIZE_N, D), dtype=tl.float32)

        for block_q in range(block_kv, len_sample, BLOCK_SIZE_N):  # load Q_i, dQ_i, O_i, dO_i, d_attn_i
            q = tl.zeros((BLOCK_SIZE_N, D), dtype=tl.float32)
            q_block_ptrs = tl.make_block_ptr(
                base=Q_ptr + pid_h * stride_qh + start * stride_qn,
                shape = (len_sample.to(tl.int32), D),
                strides = (stride_qn, stride_qd),
                offsets = (block_q, 0),
                block_shape = (BLOCK_SIZE_N, D),
                order = (0, 1)
            )

            dq_block_ptrs = tl.make_block_ptr(
                base=dQ_ptr + pid_h * stride_qh + start * stride_qn,
                shape = (len_sample.to(tl.int32), D),
                strides = (stride_qn, stride_qd),
                offsets = (block_q, 0),
                block_shape = (BLOCK_SIZE_N, D),
                order = (0, 1)
            )

            q = tl.load(q_block_ptrs)
            d_q = tl.load(dq_block_ptrs)

            o_block_ptrs = tl.make_block_ptr(
                    base = Out_ptr + pid_h*stride_out_h + start*stride_out_n,
                    shape = (len_sample.to(tl.int32) , D),
                    strides = (stride_out_n, stride_out_d),
                    offsets = (block_q , 0), #k_i (N,D) * q_j.T (D, N) -> o_ji (N, N)
                    block_shape = (BLOCK_SIZE_N, D),
                    order = (0, 1)
                )
            do_block_ptrs = tl.make_block_ptr(
                    base = dOut_ptr + pid_h*stride_out_h + start*stride_out_n,
                    shape = (len_sample.to(tl.int32) , D),
                    strides = (stride_out_n, stride_out_d),
                    offsets = (block_q , 0), #k_i (N,D) * q_j.T (D, N) -> o_ji (N, N)
                    block_shape = (BLOCK_SIZE_N, D),
                    order = (0, 1)
                )
            
            #o = tl.load(o_block_ptrs)
            d_o = tl.load(do_block_ptrs)  # (N, D)

            rab_ptrs = tl.make_block_ptr(  # rab shape : (B,1,N,N)
                base = rab_ptr + pid_b * stride_rab_b,
                shape = (N, N),
                strides = (stride_rab_n, stride_rab_m),
                offsets = (block_q , block_kv ),
                block_shape = (BLOCK_SIZE_N, BLOCK_SIZE_N),
                order = (0, 1)
            )
            rab = tl.load(rab_ptrs)

            #计算qk
            qk = tl.dot(q, k.T, input_precision = "ieee") + rab
            sigmoid_qk = tl.sigmoid(qk)
            qk_normalized = (qk * sigmoid_qk) / N

            d_silu_qk = sigmoid_qk * (1 + qk * (1 - sigmoid_qk))

            d_qk = tl.dot(d_o, v.T, input_precision = "ieee") / N
            # (BLOCK_SIZE_N, D) * (D, BLOCK_SIZE_N) -> (BLOCK_SIZE_N, BLOCK_SIZE_N)

                
            if block_kv == block_q:  #mask的处理方式
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
                qk_normalized = qk_normalized * attn_mask  #(BLOCK_SIZE_N, BLOCK_SIZE_N)
                d_qk = d_qk * attn_mask #掩码梯度

            d_v += tl.dot(qk_normalized.T, d_o, input_precision = "ieee")
            #mask_qk @ d_o  (m, n)@(n, d) -> (m, d)

            d_qk = d_qk * d_silu_qk

            d_q += tl.dot(d_qk, k, input_precision = "ieee") 
            # (BLOCK_SIZE_N, BLOCK_SIZE_N) * (BLOCK_SIZE_N, D) -> (BLOCK_SIZE_N, D)

            d_k += tl.dot(d_qk.T, q, input_precision = "ieee")

            tl.store(dq_block_ptrs, d_q)


        tl.store(dv_block_ptrs, d_v)
        tl.store(dk_block_ptrs, d_k)


def fused_jagged_hstu_backward(d_attn, attn,  q, k, v, rab, attn_mask, head, dim, n, x_offsets):
    # d_attn : (sum_N, num_heads*d)
    sum_N, _ = d_attn.shape
    d_attn = d_attn.view(sum_N, head, dim).permute(1, 0, 2).contiguous()
    attn = attn.view(sum_N, head, dim).permute(1, 0, 2).contiguous()
    q = q.view(sum_N, head, dim).permute(1, 0, 2).contiguous() # (head, sum_N, d)
    k = k.view(sum_N, head, dim).permute(1, 0, 2).contiguous() # (head, sum_N, d)
    v = v.view(sum_N, head, dim).permute(1, 0, 2).contiguous() # (head, sum_N, d

    B = x_offsets.shape[0] - 1
    d_q = torch.zeros_like(q)
    d_k = torch.zeros_like(k)
    d_v = torch.zeros_like(v)

    grid = (head, B)

    fused_backward_kernel[grid](
        q, k, v, rab,
        d_q, d_k, d_v,
        attn,
        d_attn ,
        attn_mask,
        x_offsets,
        B, head, n, dim,
        k.stride(0), k.stride(1), k.stride(2),
        q.stride(0), q.stride(1), q.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        rab.stride(0), rab.stride(1), rab.stride(2), rab.stride(3),
        attn_mask.stride(2), attn_mask.stride(3),
        attn.stride(0), attn.stride(1), attn.stride(2),
    )
    return d_q, d_k, d_v



