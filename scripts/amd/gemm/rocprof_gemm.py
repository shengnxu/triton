#!/usr/bin/env python3
import argparse
import sys

import torch
import gemm_kernels

def test_gemm(M, N, K, dtype, verbose = False):
    a = torch.randn((M, K), device='cuda', dtype=dtype)
    b = torch.randn((K, N), device='cuda', dtype=dtype)
    c = gemm_kernels.matmul(a, b)

    if verbose:
        best_config = gemm_kernels.get_best_config(M, N, K)
        print(f'best_tuning_config = {best_config}')

    return c


def main(args=None):
    if args is None:
        args = sys.argv[1:]

    parser = argparse.ArgumentParser(
        prog="test gemm tuning",
        description="Tuning infra for triton gemm",
        allow_abbrev=False,
    )

    parser.add_argument("-m", type=int, default=argparse.SUPPRESS)
    parser.add_argument("-n", type=int, default=argparse.SUPPRESS)
    parser.add_argument("-k", type=int, default=argparse.SUPPRESS)
    parser.add_argument("-dtype", type=str, default='fp16', help="Input/output data type")
    parser.add_argument("-v", action='store_true', default=False, help="Print best tuning config, default not print")
    parsed_args = parser.parse_args(args)

    dtype = torch.float16
    if parsed_args.dtype == 'fp16':
        dtype = torch.float16
    elif parsed_args.dtype == 'fp32':
        dtype = torch.float32
    elif parsed_args.dtype == 'bf16':
        dtype = torch.bfloat16
    else:
        print(f"Unsupported datatype {args.dtype}")
        sys.exit(1)
    print(f"dtype = {parsed_args.dtype}")

    verbose = parsed_args.v

    M = parsed_args.m
    N = parsed_args.n
    K = parsed_args.k
    test_gemm(M, N, K, dtype, verbose)


if __name__ == '__main__':
    sys.exit(main())
