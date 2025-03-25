# 经过测试，由于要重复计算指针，速度不如v1

import torch
import triton
import triton.language as tl
import torch.nn.functional as F

@triton.jit
def silu(x):
    return x*tl.sigmoid(x)  #和用F.silu会导致平均4.46的误差？


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_N": 32}, num_warps=4,num_stages=3),
        #triton.Config({"BLOCK_SIZE_N": 128}, num_warps=8),
    ],
    key=["N"],
)

@triton.jit
def hstu_fused_attention_kernel_rab(
    Q_ptr, K_ptr, V_ptr, rab_ptr,
    Out_ptr,
    B, H, N, D :tl.constexpr,
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_rab_b, stride_rab_h,stride_rab_n, stride_rab_m,
    stride_out_b, stride_out_h, stride_out_n, stride_out_d,
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
        #加载K和V的块， K_i V_i
        k = tl.load(k_ptrs)
        v = tl.load(v_ptrs)
        
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
                    shape = (N,D),
                    strides = (stride_out_n, stride_out_d),
                    offsets = (block_q, 0), #k_i (N,D) * q_j.T (D, N) -> o_ji (N, N)
                    block_shape = (BLOCK_SIZE_N, D),
                    order = (0, 1)
            )
            #tl.static_print("o_ptrs shape", o_ptrs)
            #加载Q的块 Q_j
            q = tl.load(q_ptrs)
            #加载output的块  O_j
            o = tl.load(o_ptrs)
            #计算Q_j * K_i, 得到QK_ji, (BLOCK_N, BLOCK_N)

            rab_ptrs = tl.make_block_ptr(
                base = rab_ptr + pid_b*stride_rab_b,
                shape = (N, N),
                strides = (stride_rab_n, stride_rab_m),
                offsets = (block_q, block_kv),
                block_shape = (BLOCK_SIZE_N, BLOCK_SIZE_N),
                order = (0, 1)
            )
            rab = tl.load(rab_ptrs)
            qk = silu(tl.dot(q, k.T,input_precision = "ieee") + rab)/N

            #经测试，triton的qk与einsum的qk误差在3.45e-11
            attn = tl.dot(qk, v,input_precision = "ieee")
            o += attn
            tl.store(o_ptrs, o)




@triton.jit
def hstu_fused_attention_kernel_norab(
    Q_ptr, K_ptr, V_ptr,
    Out_ptr,
    B, H, N, D :tl.constexpr,
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_out_b, stride_out_h, stride_out_n, stride_out_d,
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
        #加载K和V的块， K_i V_i
        k = tl.load(k_ptrs)
        v = tl.load(v_ptrs)
        
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
                    shape = (N,D),
                    strides = (stride_out_n, stride_out_d),
                    offsets = (block_q, 0), #k_i (N,D) * q_j.T (D, N) -> o_ji (N, N)
                    block_shape = (BLOCK_SIZE_N, D),
                    order = (0, 1)
            )
            #tl.static_print("o_ptrs shape", o_ptrs)
            #加载Q的块 Q_j
            q = tl.load(q_ptrs)
            #加载output的块  O_j
            o = tl.load(o_ptrs)
            #计算Q_j * K_i, 得到QK_ji, (BLOCK_N, BLOCK_N)
            qk = silu(tl.dot(q, k.T,input_precision = "ieee"))/N

            #经测试，triton的qk与einsum的qk误差在3.45e-11
            attn = tl.dot(qk, v,input_precision = "ieee")
            o += attn
            tl.store(o_ptrs, o)


def hstu_fused_attention(q, k, v, rab, enable_rab):  #N为padded后的长度， 输入的q, k 形状为[B, N, H*D], v形状为[B, N, H*D]
    B,N,H,D = q.shape
    
    q = q.permute(0,2,1,3).contiguous() #[B, N, H, D] -> [B, H, N, D]
    k = k.permute(0,2,1,3).contiguous()
    v = v.permute(0,2,1,3).contiguous()
    rab = rab.contiguous()

    # 预分配输出张量
    output = torch.zeros_like(q)

    # 调用 Triton 内核
    grid = (B, H)  # 每个batch的每个head单独计算
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    

    # 记录开始时间
    start_event.record()

    if enable_rab:
        hstu_fused_attention_kernel_rab[grid](
            q, k, v, rab, output,
            B, H, N, D,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            rab.stride(0), rab.stride(1), rab.stride(2), rab.stride(3),
            output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        )
    else:
        hstu_fused_attention_kernel_norab[grid](
            q, k, v, output,
            B, H, N, D,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        )
    end_event.record()
    torch.cuda.synchronize()
    print("Triton Time: ", start_event.elapsed_time(end_event))
    
    return output