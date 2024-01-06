import torch

import triton
import triton.language as tl
import triton.ops.blocksparse as tlb

Z,H,M,N,K = 4, 2, 256 , 512, 384

a = torch.rand((Z, H, M, K), dtype=torch.float32).cuda()
b = torch.rand((Z, H, K, N), dtype=torch.float32).cuda()
block = 16
layout = torch.randint(0, 2, (H, M//block, N//block))

dot = tlb.matmul(layout, block, 'sdd', a.device, trans_a=False, trans_b=False)
c = dot(a, b)

print('a.shape = ', a.shape)
print('c.shape = ', c.shape)
print('layout.shape = ', layout.shape)
print('---------------------------')
print('finding global_load for softmax')
softmax = tlb.softmax(layout, block, a.device, is_dense=True)
d = softmax(c, scale=1.0, rel_logits=a)

