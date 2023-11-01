#!/usr/bin/python3
import torch
from torch.testing import assert_close

import triton
import triton._C.libtriton.triton as _triton
import triton.language as tl
from triton.runtime.jit import JITFunction, TensorWrapper, reinterpret
import triton.ops as to
import traceback
import argparse

@triton.jit
def add_kernel(
    x_ptr,  # *Pointer* to first input vector.
    y_ptr,  # *Pointer* to second input vector.
    output_ptr,  # *Pointer* to output vector.
    n_elements,  # Size of the vector.
    BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
                 # NOTE: `constexpr` so it can be used as a shape value.
):
    # There are multiple 'programs' processing different data. We identify which program
    # we are here:
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    # This program will process inputs that are offset from the initial data.
    # For instance, if you had a vector of length 256 and block_size of 64, the programs
    # would each access the elements [0:64, 64:128, 128:192, 192:256].
    # Note that offsets is a list of pointers:
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create a mask to guard memory operations against out-of-bounds accesses.
    mask = offsets < n_elements
    # Load x and y from DRAM, masking out any extra elements in case the input is not a
    # multiple of the block size.
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    # Write x + y back to DRAM.
    tl.store(output_ptr + offsets, output, mask=mask)

def vecadd(a: torch.tensor, b: torch.tensor):
    assert a.shape[0] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert b.is_contiguous(), "Matrix B must be contiguous"

    # Allocates output.
    c = torch.empty_like(b, dtype=torch.float32)
    n_elements = c.numel()
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (
        triton.cdiv(n_elements, META['BLOCK_SIZE']),)
    add_kernel[grid](
        a, b, c,
        n_elements, BLOCK_SIZE = 1024,
    )

    return c

def test_vec_add(SIZE, ab_type, a_is_f8 = False):

    @triton.jit
    def copy_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        offsets = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        input = tl.load(input_ptr + offsets, mask=mask)
        output = input
        tl.store(output_ptr + offsets, output, mask=mask)

    print("testing sizes: SIZE: {}, ab type: {}, a_is_f8: {}".format(SIZE, ab_type, a_is_f8))

    if a_is_f8:
        f8_tensor = torch.randn((SIZE,), dtype=torch.float32, device='cuda') * 10
        # f8_tensor = f8_tensor.to(torch.int8)
        # # f32_to_f8 doesn't handle nan, so we make sure f8_tensor doesn't contain any nan
        # all_exp_ones = (f8_tensor & 0b01111100) == 128 - 2**a_type.fp_mantissa_width
        # f8_tensor[all_exp_ones] = 0
        a = f8_tensor.to(torch.float8_e5m2fnuz)
        a_f16 = a.to(torch.float32)

        # n_elements = f8_tensor.numel()
        # # grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        # a_f16 = torch.empty_like(f8_tensor, dtype=torch.float32)
        # copy_kernel[grid](triton.reinterpret(f8_tensor, a_type), a_f16, n_elements, BLOCK_SIZE=1024)
        b_f16 = torch.randn((SIZE,), device = 'cuda', dtype=torch.float32)

        print(f'a = {a_f16}')
        print(f'b = {b_f16}')
        # print(f'f8_a = {f8_tensor}')
        golden = torch.add(a_f16, b_f16)
        c = vecadd(a, b_f16)

        print(f'gold = {golden}')
        print(f'c = {c}')

    else:
        a = torch.randn((SIZE,), device='cuda', dtype=ab_type)
        b = torch.randn((SIZE,), device='cuda', dtype=ab_type)
        golden = torch.add(a, b)
        c = vecadd(a, b)
        print(f'gold = {golden}')
        print(f'c = {c}')
    
    # torch.set_printoptions(profile="full")
    torch.testing.assert_close(c, golden, atol=1e-3, rtol=1e-3)
    golden_abs_err = 0.5
    golden_rel_err = 0.0
    # assert_close(c.to(torch.float64), golden.to(torch.float64), rtol=max(1e-3, 10 * golden_rel_err), atol=max(1e-3, 10 * golden_abs_err), check_dtype=False)

def parse_arguments():
    parser = argparse.ArgumentParser(
        prog = "test gemm tuning",
        description= "Tuning MFMA GEMM implementation",
        allow_abbrev=False,
    )

    parser.add_argument("-size", type=int, required=True, default=argparse.SUPPRESS)
    parser.add_argument("--fp8", action='store_true', default=False)

    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    SIZE = args.size
    a_is_fp8 = False
    if args.fp8:
        a_is_fp8 = True

    SIZE = 2 ** SIZE
    print(f'element_num = {SIZE}')

    try:
        test_vec_add(SIZE, torch.float16, a_is_f8 = a_is_fp8)
    except:
        traceback.print_exc()
        print("FAILED!")
    else:
        print("PASSED!")

if __name__ == "__main__":
    main()
