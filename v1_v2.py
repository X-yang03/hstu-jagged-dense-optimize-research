import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
import random
import fbgemm_gpu
import torch.profiler
from fused_jagged_hstu.fused_hstu_op import FusedHSTUOp
from fused_jagged_hstu.fused_simpler_op import FusedHSTUOp_
from fused_hstu_v2.fused_hstu_op_v2 import FusedHSTUOpv2
from fused_hstu_v3.fused_hstu_op_v3 import FusedHSTUOpv3

def get_input(sum_N, head, d, B, n):
    q = torch.randn(sum_N, head*d, requires_grad=True, device="cuda")
    k = torch.randn(sum_N, head*d, requires_grad=True, device="cuda")
    v = torch.randn(sum_N, head*d, requires_grad=True, device="cuda")
    rab = torch.randn(B, 1, n, n, requires_grad=True, device="cuda")

    q1 = q.clone().detach().requires_grad_(True)
    k1 = k.clone().detach().requires_grad_(True)
    v1 = v.clone().detach().requires_grad_(True)
    rab1 = rab.clone().detach().requires_grad_(True)
    
    # 生成一个下三角矩阵
    attn_mask = torch.tril(torch.ones((n, n), device='cuda:0'))
    # 调整形状为 (1, 1, n, n)
    attn_mask = attn_mask.view(1, 1, n, n) 
    return q, k, v, rab,  q1, k1, v1, rab1, attn_mask

seq_len = [128, 120, 256, 260, 512, 510, 1024, 1020, 100, 200, 300, 400]
max_seq = 200
min_seq = 100
n = 0
B = 20
x_offsets = [0]
for i in range(1, B+1):
    rand_seq_len = random.choice(seq_len)
    # rand_seq_len = random.randint(min_seq, max_seq)
    n = max(n, rand_seq_len)
    x_offsets.append(x_offsets[-1] + rand_seq_len) # 生成一个长度为B的序列，每个元素为0-1024之间的随机数
x_offsets = torch.tensor(x_offsets, device="cuda") # 转换为tensor

n += 11
head, d = 2 , 25
sum_N = int(x_offsets[-1])

print('benchmark config: sum_N: {}, head: {}, d: {}, B: {}, n: {}'.format(sum_N, head, d, B, n))
print('input q k v & output shape: ({}, {})'.format(sum_N, head*d))
print('input rab shape: ({}, {}, {}, {})'.format(B, 1, n, n))
print('input attn_mask shape: ({}, {}, {}, {})'.format(1, 1, n, n))

print('===========================================================')

print('warm up')
for _ in tqdm(range(3)):
    q, k, v, rab, q1, k1, v1, rab1, attn_mask = get_input(sum_N, head, d, B, n)
    warmup_v1_attn = FusedHSTUOpv3.apply(q, k, v, rab, attn_mask, head, d, n, x_offsets)
    #warmup_v1_attn = v2HSTUOp_.apply(q, k, v, rab, attn_mask, head, d, n, x_offsets)
    loss = warmup_v1_attn.sum()
    loss.backward()
    warmup_v2_attn = FusedHSTUOpv2.apply(q1, k1, v1, rab1, attn_mask, head, d, n, x_offsets)
    loss1 = warmup_v2_attn.sum()
    loss1.backward()
print('warm up done')

print('===========================================================')

print('start benchmark')

v1_forward_time = []
v2_forward_time = []
v1_backward_time = []
v2_backward_time = []

test_num = 10
for _ in tqdm(range(test_num)):
    q, k, v, rab, q1, k1, v1, rab1, attn_mask = get_input(sum_N, head, d, B, n)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)


    start_event.record()
    v1_attn = FusedHSTUOpv3.apply(q, k, v, rab, attn_mask, head, d, n, x_offsets)
    # v1_attn = v2HSTUOp_.apply(q, k, v, rab, attn_mask, head, d, n, x_offsets)
    end_event.record()
    torch.cuda.synchronize()
    v1_forward_time.append(start_event.elapsed_time(end_event))

    start_event.record()
    v2_attn = FusedHSTUOpv2.apply(q1, k1, v1, rab, attn_mask, head, d, n, x_offsets)
    end_event.record()
    torch.cuda.synchronize()
    v2_forward_time.append(start_event.elapsed_time(end_event))

    attn_true = torch.randn_like(v1_attn)

    criterion = nn.CrossEntropyLoss()
    loss = criterion(v1_attn, attn_true)
    start_event.record()
    loss.backward()
    end_event.record()
    torch.cuda.synchronize()
    v1_backward_time.append(start_event.elapsed_time(end_event))

    criterion1 = nn.CrossEntropyLoss()
    loss1 = criterion1(v2_attn, attn_true)
    start_event.record()
    loss1.backward()
    end_event.record()
    torch.cuda.synchronize()
    v2_backward_time.append(start_event.elapsed_time(end_event))

print("avg v1 forward time: ", sum(v1_forward_time) / len(v1_forward_time))
print("avg v2 forward time: ", sum(v2_forward_time) / len(v2_forward_time))
print("avg v1 backward time: ", sum(v1_backward_time) / len(v1_backward_time))
print("avg v2 backward time: ", sum(v2_backward_time) / len(v2_backward_time))

print('===========================================================')

speedup_forward = [v1_forward_time[i] / v2_forward_time[i] for i in range(len(v1_forward_time))]
speedup_backward = [v1_backward_time[i] / v2_backward_time[i] for i in range(len(v1_backward_time))]

print("avg forward speedup: ", sum(speedup_forward) / len(speedup_forward))
print("avg backward speedup: ", sum(speedup_backward) / len(speedup_backward))
print('===========================================================')
print('benchmark done')

print(v1_backward_time)
print(v2_backward_time)