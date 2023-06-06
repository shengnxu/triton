#!/usr/bin/python3
import torch
from torch.testing import assert_close

import triton
import triton.language as tl
import triton.ops as to
import traceback

def get_variant_golden(a, b):
    SIZE_M = a.shape[0]
    SIZE_K = a.shape[1]
    SIZE_N = b.shape[1]
    assert a.shape[1] == b.shape[0]
    zero_M_K = torch.zeros((SIZE_M, SIZE_K)).cuda()
    zero_3M_K = torch.zeros((3 * SIZE_M, SIZE_K)).cuda()
    zero_K_N = torch.zeros((SIZE_K, SIZE_N)).cuda()
    zero_3K_N = torch.zeros((3 * SIZE_K, SIZE_N)).cuda()
    a_padded = torch.cat((a, zero_M_K, zero_M_K), 0)
    a_padded = torch.cat((a_padded, zero_3M_K, zero_3M_K), 1)
    b_padded = torch.cat((b, zero_K_N, zero_K_N), 0)
    b_padded = torch.cat((b_padded, zero_3K_N, zero_3K_N), 1)
    c_padded = torch.matmul(a_padded, b_padded)
    return c_padded[:SIZE_M, :SIZE_N]

def test_gemm(SIZE_M, SIZE_N, SIZE_K, ab_type, c_type):
    print("testing sizes: M: {}, N: {}, K: {}, ab type: {}, c type: {}".format(SIZE_M, SIZE_N, SIZE_K, ab_type, c_type))
    if ab_type == torch.int8:
        a = torch.randn((SIZE_M, SIZE_K), device='cuda', dtype=torch.float32).to(torch.int8)
        b = torch.randn((SIZE_K, SIZE_N), device='cuda', dtype=torch.float32).to(torch.int8)
        golden = torch.matmul(a.to(torch.float64), b.to(torch.float64))
    
        golden_abs_err = 0.5
        golden_rel_err = 0.0
    else:
        a = torch.randn((SIZE_M, SIZE_K), device='cuda', dtype=ab_type)
        b = torch.randn((SIZE_K, SIZE_N), device='cuda', dtype=ab_type)
        golden = torch.matmul(a, b)
    
        golden_variant = get_variant_golden(a, b)
        golden_diff = golden - golden_variant
        golden_abs_err = torch.max(torch.abs(golden_diff)).item()
        golden_rel_err = torch.max(torch.abs(golden_diff / golden)).item()

    c = to.matmul(a, b, c_type)

    torch.set_printoptions(profile="full")
    assert_close(c.to(torch.float64), golden.to(torch.float64), rtol=max(1e-3, 10 * golden_rel_err), atol=max(1e-3, 10 * golden_abs_err), check_dtype=False)

# configs=[
# (1104,1,4608),
# (1104,16,4608),
# (1104,1335,4608),
# (1104,1408,4608),
# (4608,1,320),
# (4608,16,320),
# (4608,1335,320),
# (4608,1408,320),
# (16,1,4608),
# (16,16,4608),
# (16,1335,4608),
# (16,1408,4608),
# (768,1,4608),
# (768,16,4608),
# (768,1335,4608),
# (768,1408,4608),
# (4608,1,768),
# (4608,16,768),
# (4608,1335,768),
# (4608,1408,768),
# ]

# configs=[
# (2208,1,4608),
# (2208,2,4608),
# (2208,4,4608),
# (2208,8,4608),
# (2208,16,4608),
# (2208,1335,4608),
# (4608,1,320),
# (4608,2,320),
# (4608,4,320),
# (4608,8,320),
# (4608,16,320),
# (4608,1335,320),
# (16,1,4608),
# (16,2,4608),
# (16,4,4608),
# (16,8,4608),
# (16,16,4608),
# (16,1335,4608),
# (1536,1,4608),
# (1536,16,4608),
# (1536,1335,4608),
# (4608,1,1536),
# (4608,16,1536),
# (4608,1335,1536),
# ]

configs=[
    (32, 32, 32),
    (32, 1, 32),
    (1, 32, 32),
    (32, 32, 1),
    (1104, 1, 4608),
    (1104, 2, 4608),
    (1104, 4, 4608),
    (1104, 8, 4608),
    (1104, 16, 4608),
    (1104, 1335, 4608),
    (4608, 1, 320),
    (4608, 2, 320),
    (4608, 4, 320),
    (4608, 8, 320),
    (4608, 16, 320),
    (4608, 1335, 320),
    (16, 1, 4608),
    (16, 2, 4608),
    (16, 4, 4608),
    (16, 8, 4608),
    (16, 16, 4608),
    (16, 1335, 4608),
    (768, 1, 4608),
    (768, 16, 4608),
    (768, 1335, 4608),
    (4608, 1, 768),
    (4608, 16, 768),
    (4608, 1335, 768),
]

for cfg in configs:
    try:
        test_gemm(*cfg, torch.float16, torch.float32)
    except:
        traceback.print_exc()
        print("FAILED!")
    else:
        print("PASSED!")