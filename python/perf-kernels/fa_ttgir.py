
"""
Fused Attention
===============

This is a Triton implementation of the Flash Attention v2 algorithm from Tri Dao (https://tridao.me/publications/flash2/flash2.pdf)

Extra Credits:
- Original flash attention paper (https://arxiv.org/abs/2205.14135)
- Rabe and Staats (https://arxiv.org/pdf/2112.05682v2.pdf)
- Adam P. Goucher for simplified vector math

"""

import pytest
import torch
import sys

import triton
import triton.language as tl


ir = """
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [32, 2], warpsPerCTA = [4, 1], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [4, 16], warpsPerCTA = [1, 4], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#mfma = #triton_gpu.mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>
#shared = #triton_gpu.shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = false}>
#shared1 = #triton_gpu.shared<{vec = 4, perPhase = 2, maxPhase = 4, order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = false}>
module attributes {"triton_gpu.compute-capability" = 0 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {
  tt.func public @_attn_fwd_0d1d2d34d5d6de7de8de9c10de11de12de13c14de15de16de17c18de19de20de21c2223de24de(%arg0: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg3: f32, %arg4: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg7: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg8: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg9: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg10: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg11: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg12: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg13: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg14: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg15: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg16: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg17: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg18: i32, %arg19: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg20: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<1.000000e+00> : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
    %cst_0 = arith.constant dense<0xFF800000> : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mfma>
    %c128_i64 = arith.constant 128 : i64
    %c0_i64 = arith.constant 0 : i64
    %c128_i32 = arith.constant 128 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst_2 = arith.constant 1.44269502 : f32
    %0 = arith.mulf %arg3, %cst_2 : f32
    %1 = tt.splat %0 : (f32) -> tensor<128x128xf32, #blocked>
    %2 = tt.get_program_id x : i32
    %3 = arith.muli %2, %c128_i32 : i32
    %4 = arith.extsi %3 : i32 to i64
    %5 = tt.splat %4 : (i64) -> tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %6 = tt.splat %4 : (i64) -> tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
    %7 = tt.get_program_id y : i32
    %8 = arith.muli %7, %arg7 : i32
    %9 = tt.addptr %arg0, %8 : !tt.ptr<f16, 1>, i32
    %10 = tt.splat %9 : (!tt.ptr<f16, 1>) -> tensor<128x1x!tt.ptr<f16, 1>, #blocked>
    %11 = arith.extsi %arg8 : i32 to i64
    %12 = tt.splat %11 : (i64) -> tensor<128x1xi64, #blocked>
    %13 = tt.addptr %arg1, %8 : !tt.ptr<f16, 1>, i32
    %14 = arith.extsi %arg11 : i32 to i64
    %15 = tt.addptr %arg2, %8 : !tt.ptr<f16, 1>, i32
    %16 = arith.extsi %arg14 : i32 to i64
    %17 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %18 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
    %19 = arith.extsi %17 : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> to tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %20 = arith.extsi %18 : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #mfma}>> to tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
    %21 = arith.addi %5, %19 : tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %22 = arith.addi %6, %20 : tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
    %23 = tt.expand_dims %21 {axis = 1 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>>) -> tensor<128x1xi64, #blocked>
    %24 = tt.expand_dims %22 {axis = 1 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #mfma}>>) -> tensor<128x1xi64, #mfma>
    %25 = arith.muli %23, %12 : tensor<128x1xi64, #blocked>
    %26 = tt.addptr %10, %25 : tensor<128x1x!tt.ptr<f16, 1>, #blocked>, tensor<128x1xi64, #blocked>
    %27 = tt.broadcast %26 : (tensor<128x1x!tt.ptr<f16, 1>, #blocked>) -> tensor<128x128x!tt.ptr<f16, 1>, #blocked>
    %28 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %29 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #mfma}>>
    %30 = arith.extsi %28 : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> to tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %31 = arith.extsi %29 : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #mfma}>> to tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #mfma}>>
    %32 = tt.expand_dims %30 {axis = 0 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked}>>) -> tensor<1x128xi64, #blocked>
    %33 = tt.expand_dims %31 {axis = 0 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #mfma}>>) -> tensor<1x128xi64, #mfma>
    %34 = tt.broadcast %32 : (tensor<1x128xi64, #blocked>) -> tensor<128x128xi64, #blocked>
    %35 = tt.broadcast %33 : (tensor<1x128xi64, #mfma>) -> tensor<128x128xi64, #mfma>
    %36 = tt.addptr %27, %34 : tensor<128x128x!tt.ptr<f16, 1>, #blocked>, tensor<128x128xi64, #blocked>
    %65 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %66 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %67 = arith.extsi %65 : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>> to tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %68 = arith.extsi %66 : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>> to tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %69 = tt.expand_dims %67 {axis = 0 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>) -> tensor<1x128xi64, #blocked1>
    %70 = tt.expand_dims %68 {axis = 1 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>) -> tensor<128x1xi64, #blocked1>
    %71 = tt.splat %13 : (!tt.ptr<f16, 1>) -> tensor<128x1x!tt.ptr<f16, 1>, #blocked1>
    %72 = tt.addptr %71, %70 : tensor<128x1x!tt.ptr<f16, 1>, #blocked1>, tensor<128x1xi64, #blocked1>
    %73 = tt.broadcast %72 : (tensor<128x1x!tt.ptr<f16, 1>, #blocked1>) -> tensor<128x128x!tt.ptr<f16, 1>, #blocked1>
    %74 = tt.splat %14 : (i64) -> tensor<1x128xi64, #blocked1>
    %75 = arith.muli %69, %74 : tensor<1x128xi64, #blocked1>
    %76 = tt.broadcast %75 : (tensor<1x128xi64, #blocked1>) -> tensor<128x128xi64, #blocked1>
    %77 = tt.addptr %73, %76 : tensor<128x128x!tt.ptr<f16, 1>, #blocked1>, tensor<128x128xi64, #blocked1>


    %37 = triton_gpu.view_slice %36[0, 96] [128, 32] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked> to tensor<128x32x!tt.ptr<f16, 1>, #blocked>
    %38 = triton_gpu.view_slice %36[0, 64] [128, 32] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked> to tensor<128x32x!tt.ptr<f16, 1>, #blocked>
    %39 = triton_gpu.view_slice %36[0, 32] [128, 32] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked> to tensor<128x32x!tt.ptr<f16, 1>, #blocked>
    %40 = triton_gpu.view_slice %36[0, 0] [128, 32] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked> to tensor<128x32x!tt.ptr<f16, 1>, #blocked>

    %49 = triton_gpu.view_slice %1[0, 96] [128, 32] [1, 1] : tensor<128x128xf32, #blocked> to tensor<128x32xf32, #blocked>
    %50 = triton_gpu.view_slice %1[0, 64] [128, 32] [1, 1] : tensor<128x128xf32, #blocked> to tensor<128x32xf32, #blocked>
    %51 = triton_gpu.view_slice %1[0, 32] [128, 32] [1, 1] : tensor<128x128xf32, #blocked> to tensor<128x32xf32, #blocked>
    %52 = triton_gpu.view_slice %1[0, 0] [128, 32] [1, 1] : tensor<128x128xf32, #blocked> to tensor<128x32xf32, #blocked>

    %81 = triton_gpu.view_slice %77[0, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked1> to tensor<32x128x!tt.ptr<f16, 1>, #blocked1>
    %85 = tt.load %81 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #blocked1>
    %78 = triton_gpu.view_slice %77[32, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked1> to tensor<32x128x!tt.ptr<f16, 1>, #blocked1>
    %82 = tt.load %78 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #blocked1>

    %44 = tt.load %40 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x32xf16, #blocked>
    %48 = arith.extf %44 : tensor<128x32xf16, #blocked> to tensor<128x32xf32, #blocked>
    %56 = arith.mulf %48, %52 : tensor<128x32xf32, #blocked>
    %60 = arith.truncf %56 : tensor<128x32xf32, #blocked> to tensor<128x32xf16, #blocked>
    %64 = triton_gpu.convert_layout %60 : (tensor<128x32xf16, #blocked>) -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 8}>>

    %43 = tt.load %39 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x32xf16, #blocked>
    %47 = arith.extf %43 : tensor<128x32xf16, #blocked> to tensor<128x32xf32, #blocked>
    %55 = arith.mulf %47, %51 : tensor<128x32xf32, #blocked>
    %59 = arith.truncf %55 : tensor<128x32xf32, #blocked> to tensor<128x32xf16, #blocked>
    %63 = triton_gpu.convert_layout %59 : (tensor<128x32xf16, #blocked>) -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 8}>>
    %89 = triton_gpu.convert_layout %85 : (tensor<32x128xf16, #blocked1>) -> tensor<32x128xf16, #shared>
    gpu.barrier
    %93 = triton_gpu.convert_layout %89 : (tensor<32x128xf16, #shared>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 8}>>
    %94 = tt.dot %64, %93, %cst_1 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 8}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 8}>> -> tensor<128x128xf32, #mfma>
    gpu.barrier

    %86 = triton_gpu.convert_layout %82 : (tensor<32x128xf16, #blocked1>) -> tensor<32x128xf16, #shared>
    %79 = triton_gpu.view_slice %77[64, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked1> to tensor<32x128x!tt.ptr<f16, 1>, #blocked1>
    %83 = tt.load %79 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #blocked1>
    %42 = tt.load %38 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x32xf16, #blocked>
    %46 = arith.extf %42 : tensor<128x32xf16, #blocked> to tensor<128x32xf32, #blocked>
    %54 = arith.mulf %46, %50 : tensor<128x32xf32, #blocked>
    %58 = arith.truncf %54 : tensor<128x32xf32, #blocked> to tensor<128x32xf16, #blocked>
    %62 = triton_gpu.convert_layout %58 : (tensor<128x32xf16, #blocked>) -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 8}>>
    gpu.barrier
    %90 = triton_gpu.convert_layout %86 : (tensor<32x128xf16, #shared>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 8}>>
    %95 = tt.dot %63, %90, %94 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 8}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 8}>> -> tensor<128x128xf32, #mfma>
    gpu.barrier
    %87 = triton_gpu.convert_layout %83 : (tensor<32x128xf16, #blocked1>) -> tensor<32x128xf16, #shared>
    %80 = triton_gpu.view_slice %77[96, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked1> to tensor<32x128x!tt.ptr<f16, 1>, #blocked1>
    %84 = tt.load %80 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #blocked1>
    %41 = tt.load %37 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x32xf16, #blocked>
    %45 = arith.extf %41 : tensor<128x32xf16, #blocked> to tensor<128x32xf32, #blocked>
    %53 = arith.mulf %45, %49 : tensor<128x32xf32, #blocked>
    %57 = arith.truncf %53 : tensor<128x32xf32, #blocked> to tensor<128x32xf16, #blocked>
    %61 = triton_gpu.convert_layout %57 : (tensor<128x32xf16, #blocked>) -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 8}>>
    gpu.barrier

    %91 = triton_gpu.convert_layout %87 : (tensor<32x128xf16, #shared>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 8}>>
    %96 = tt.dot %62, %91, %95 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 8}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 8}>> -> tensor<128x128xf32, #mfma>
    gpu.barrier
    %88 = triton_gpu.convert_layout %84 : (tensor<32x128xf16, #blocked1>) -> tensor<32x128xf16, #shared>
    gpu.barrier
    %92 = triton_gpu.convert_layout %88 : (tensor<32x128xf16, #shared>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 8}>>
    %97 = tt.dot %61, %92, %96 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 8}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 8}>> -> tensor<128x128xf32, #mfma>

    %98 = tt.splat %15 : (!tt.ptr<f16, 1>) -> tensor<128x1x!tt.ptr<f16, 1>, #blocked1>
    %99 = tt.splat %16 : (i64) -> tensor<1x128xi64, #blocked1>
    %100 = arith.muli %69, %99 : tensor<1x128xi64, #blocked1>
    %101 = tt.broadcast %100 : (tensor<1x128xi64, #blocked1>) -> tensor<128x128xi64, #blocked1>
    %102:6 = scf.for %arg21 = %c0_i32 to %arg20 step %c128_i32 iter_args(%arg22 = %97, %arg23 = %cst_1, %arg24 = %cst, %arg25 = %cst_0, %arg26 = %c0_i64, %arg27 = %c0_i64) -> (tensor<128x128xf32, #mfma>, tensor<128x128xf32, #mfma>, tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>, tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>, i64, i64)  : i32 {
      %115 = arith.cmpi ne, %arg21, %c0_i32 : i32
      %116 = scf.if %115 -> (tensor<128x128xf32, #mfma>) {
        %168 = tt.splat %arg27 : (i64) -> tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
        %169 = arith.addi %168, %67 : tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
        %170 = tt.expand_dims %169 {axis = 0 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>) -> tensor<1x128xi64, #blocked1>
        %171 = arith.muli %170, %74 : tensor<1x128xi64, #blocked1>
        %172 = tt.broadcast %171 : (tensor<1x128xi64, #blocked1>) -> tensor<128x128xi64, #blocked1>
        %173 = tt.addptr %73, %172 : tensor<128x128x!tt.ptr<f16, 1>, #blocked1>, tensor<128x128xi64, #blocked1>
        %174 = triton_gpu.view_slice %173[96, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked1> to tensor<32x128x!tt.ptr<f16, 1>, #blocked1>
        %175 = triton_gpu.view_slice %173[64, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked1> to tensor<32x128x!tt.ptr<f16, 1>, #blocked1>
        %176 = triton_gpu.view_slice %173[32, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked1> to tensor<32x128x!tt.ptr<f16, 1>, #blocked1>
        %177 = triton_gpu.view_slice %173[0, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked1> to tensor<32x128x!tt.ptr<f16, 1>, #blocked1>
        %178 = tt.load %177 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #blocked1>
        %179 = triton_gpu.convert_layout %178 : (tensor<32x128xf16, #blocked1>) -> tensor<32x128xf16, #shared>
        %180 = tt.load %176 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #blocked1>
        gpu.barrier
        %181 = triton_gpu.convert_layout %179 : (tensor<32x128xf16, #shared>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 8}>>
        %182 = tt.dot %64, %181, %cst_1 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 8}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 8}>> -> tensor<128x128xf32, #mfma>
        gpu.barrier
        %183 = triton_gpu.convert_layout %180 : (tensor<32x128xf16, #blocked1>) -> tensor<32x128xf16, #shared>
        %184 = tt.load %175 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #blocked1>
        gpu.barrier
        %185 = triton_gpu.convert_layout %183 : (tensor<32x128xf16, #shared>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 8}>>
        %186 = tt.dot %63, %185, %182 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 8}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 8}>> -> tensor<128x128xf32, #mfma>
        gpu.barrier
        %187 = triton_gpu.convert_layout %184 : (tensor<32x128xf16, #blocked1>) -> tensor<32x128xf16, #shared>
        %188 = tt.load %174 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #blocked1>
        gpu.barrier
        %189 = triton_gpu.convert_layout %187 : (tensor<32x128xf16, #shared>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 8}>>
        %190 = tt.dot %62, %189, %186 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 8}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 8}>> -> tensor<128x128xf32, #mfma>
        gpu.barrier
        %191 = triton_gpu.convert_layout %188 : (tensor<32x128xf16, #blocked1>) -> tensor<32x128xf16, #shared>
        gpu.barrier
        %192 = triton_gpu.convert_layout %191 : (tensor<32x128xf16, #shared>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 8}>>
        %193 = tt.dot %61, %192, %190 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 8}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 8}>> -> tensor<128x128xf32, #mfma>
        scf.yield %193 : tensor<128x128xf32, #mfma>
      } else {
        scf.yield %arg22 : tensor<128x128xf32, #mfma>
      }
      %117 = "tt.reduce"(%116) <{axis = 1 : i32}> ({
      ^bb0(%arg28: f32, %arg29: f32):
        %168 = arith.maximumf %arg28, %arg29 : f32
        tt.reduce.return %168 : f32
      }) : (tensor<128x128xf32, #mfma>) -> tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
      %118 = arith.maximumf %arg25, %117 : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
      %119 = tt.expand_dims %118 {axis = 1 : i32} : (tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>) -> tensor<128x1xf32, #mfma>
      %120 = tt.broadcast %119 : (tensor<128x1xf32, #mfma>) -> tensor<128x128xf32, #mfma>
      %121 = arith.subf %116, %120 : tensor<128x128xf32, #mfma>
      %122 = tt.extern_elementwise %121 {libname = "libdevice", libpath = "/triton/python/triton/language/../third_party/hip/lib/bitcode/cuda2gcn.bc", pure = true, symbol = "__nv_exp2f"} : (tensor<128x128xf32, #mfma>) -> tensor<128x128xf32, #mfma>
      %123 = arith.subf %arg25, %118 : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
      %124 = tt.extern_elementwise %123 {libname = "libdevice", libpath = "/triton/python/triton/language/../third_party/hip/lib/bitcode/cuda2gcn.bc", pure = true, symbol = "__nv_exp2f"} : (tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>) -> tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
      %125 = tt.expand_dims %124 {axis = 1 : i32} : (tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>) -> tensor<128x1xf32, #mfma>
      %126 = tt.broadcast %125 : (tensor<128x1xf32, #mfma>) -> tensor<128x128xf32, #mfma>
      %127 = arith.mulf %arg23, %126 : tensor<128x128xf32, #mfma>
      %128 = tt.splat %arg26 : (i64) -> tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
      %129 = arith.addi %128, %68 : tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
      %130 = tt.expand_dims %129 {axis = 1 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>) -> tensor<128x1xi64, #blocked1>
      %131 = tt.addptr %98, %130 : tensor<128x1x!tt.ptr<f16, 1>, #blocked1>, tensor<128x1xi64, #blocked1>
      %132 = tt.broadcast %131 : (tensor<128x1x!tt.ptr<f16, 1>, #blocked1>) -> tensor<128x128x!tt.ptr<f16, 1>, #blocked1>
      %133 = tt.addptr %132, %101 : tensor<128x128x!tt.ptr<f16, 1>, #blocked1>, tensor<128x128xi64, #blocked1>
      %134 = triton_gpu.view_slice %133[96, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked1> to tensor<32x128x!tt.ptr<f16, 1>, #blocked1>
      %135 = triton_gpu.view_slice %133[64, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked1> to tensor<32x128x!tt.ptr<f16, 1>, #blocked1>
      %136 = triton_gpu.view_slice %133[32, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked1> to tensor<32x128x!tt.ptr<f16, 1>, #blocked1>
      %137 = triton_gpu.view_slice %133[0, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked1> to tensor<32x128x!tt.ptr<f16, 1>, #blocked1>
      %139 = arith.truncf %122 : tensor<128x128xf32, #mfma> to tensor<128x128xf16, #mfma>
      %140 = triton_gpu.view_slice %139[0, 96] [128, 32] [1, 1] : tensor<128x128xf16, #mfma> to tensor<128x32xf16, #mfma>
      %141 = triton_gpu.view_slice %139[0, 64] [128, 32] [1, 1] : tensor<128x128xf16, #mfma> to tensor<128x32xf16, #mfma>
      %142 = triton_gpu.view_slice %139[0, 32] [128, 32] [1, 1] : tensor<128x128xf16, #mfma> to tensor<128x32xf16, #mfma>
      %143 = triton_gpu.view_slice %139[0, 0] [128, 32] [1, 1] : tensor<128x128xf16, #mfma> to tensor<128x32xf16, #mfma>
      %144 = triton_gpu.convert_layout %140 : (tensor<128x32xf16, #mfma>) -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>
      %145 = triton_gpu.convert_layout %141 : (tensor<128x32xf16, #mfma>) -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>
      %146 = triton_gpu.convert_layout %142 : (tensor<128x32xf16, #mfma>) -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>
      %147 = triton_gpu.convert_layout %143 : (tensor<128x32xf16, #mfma>) -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>
      %138 = tt.load %137 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #blocked1>
      %149 = tt.load %136 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #blocked1>
      %153 = tt.load %135 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #blocked1>
      %157 = tt.load %134 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #blocked1>
      gpu.barrier
      %148 = triton_gpu.convert_layout %138 : (tensor<32x128xf16, #blocked1>) -> tensor<32x128xf16, #shared1>
      %152 = triton_gpu.convert_layout %149 : (tensor<32x128xf16, #blocked1>) -> tensor<32x128xf16, #shared1>
      %156 = triton_gpu.convert_layout %153 : (tensor<32x128xf16, #blocked1>) -> tensor<32x128xf16, #shared1>
      %160 = triton_gpu.convert_layout %157 : (tensor<32x128xf16, #blocked1>) -> tensor<32x128xf16, #shared1>
      gpu.barrier
      %150 = triton_gpu.convert_layout %148 : (tensor<32x128xf16, #shared1>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>>
      %154 = triton_gpu.convert_layout %152 : (tensor<32x128xf16, #shared1>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>>
      %158 = triton_gpu.convert_layout %156 : (tensor<32x128xf16, #shared1>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>>
      %161 = triton_gpu.convert_layout %160 : (tensor<32x128xf16, #shared1>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>>

      %151 = tt.dot %147, %150, %127 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>> -> tensor<128x128xf32, #mfma>
      %155 = tt.dot %146, %154, %151 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>> -> tensor<128x128xf32, #mfma>
      %159 = tt.dot %145, %158, %155 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>> -> tensor<128x128xf32, #mfma>
      %162 = tt.dot %144, %161, %159 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>> -> tensor<128x128xf32, #mfma>
      %163 = "tt.reduce"(%122) <{axis = 1 : i32}> ({
      ^bb0(%arg28: f32, %arg29: f32):
        %168 = arith.addf %arg28, %arg29 : f32
        tt.reduce.return %168 : f32
      }) : (tensor<128x128xf32, #mfma>) -> tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
      gpu.barrier
      %164 = arith.mulf %arg24, %124 : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
      %165 = arith.addf %164, %163 : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
      %166 = arith.addi %arg26, %c128_i64 : i64
      %167 = arith.addi %arg27, %c128_i64 : i64
      scf.yield %121, %162, %165, %118, %166, %167 : tensor<128x128xf32, #mfma>, tensor<128x128xf32, #mfma>, tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>, tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>, i64, i64
    }
    %103 = tt.expand_dims %102#2 {axis = 1 : i32} : (tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>) -> tensor<128x1xf32, #mfma>
    %104 = tt.broadcast %103 : (tensor<128x1xf32, #mfma>) -> tensor<128x128xf32, #mfma>
    %105 = arith.divf %102#1, %104 : tensor<128x128xf32, #mfma>
    %106 = tt.addptr %arg5, %8 : !tt.ptr<f16, 1>, i32
    %107 = arith.extsi %arg17 : i32 to i64
    %108 = arith.truncf %105 : tensor<128x128xf32, #mfma> to tensor<128x128xf16, #mfma>
    %109 = tt.splat %107 : (i64) -> tensor<128x1xi64, #mfma>
    %110 = arith.muli %24, %109 : tensor<128x1xi64, #mfma>
    %111 = tt.splat %106 : (!tt.ptr<f16, 1>) -> tensor<128x1x!tt.ptr<f16, 1>, #mfma>
    %112 = tt.addptr %111, %110 : tensor<128x1x!tt.ptr<f16, 1>, #mfma>, tensor<128x1xi64, #mfma>
    %113 = tt.broadcast %112 : (tensor<128x1x!tt.ptr<f16, 1>, #mfma>) -> tensor<128x128x!tt.ptr<f16, 1>, #mfma>
    %114 = tt.addptr %113, %35 : tensor<128x128x!tt.ptr<f16, 1>, #mfma>, tensor<128x128xi64, #mfma>
    tt.store %114, %108 {cache = 1 : i32, evict = 1 : i32} : tensor<128x128xf16, #mfma>
    tt.return
  }
}

    """

# ir = """
# #blocked = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [32, 2], warpsPerCTA = [4, 1], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
# #blocked1 = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
# #mfma = #triton_gpu.mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>
# #shared = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 16, order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = false}>
# #shared1 = #triton_gpu.shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = false}>
# module attributes {"triton_gpu.compute-capability" = 0 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {
#   tt.func public @_attn_fwd_0d1d2d34d5d6de7de8de9c10de11de12de13c14de15de16de17c18de19de20de21c2223de24de(%arg0: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg3: f32, %arg4: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg7: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg8: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg9: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg10: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg11: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg12: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg13: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg14: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg15: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg16: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg17: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg18: i32, %arg19: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg20: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}) attributes {noinline = false} {
#     %cst = arith.constant dense<1.000000e+00> : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#     %cst_0 = arith.constant dense<0xFF800000> : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#     %cst_1 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mfma>
#     %c128_i64 = arith.constant 128 : i64
#     %c0_i64 = arith.constant 0 : i64
#     %c128_i32 = arith.constant 128 : i32
#     %c0_i32 = arith.constant 0 : i32
#     %cst_2 = arith.constant 1.44269502 : f32
#     %0 = arith.mulf %arg3, %cst_2 : f32
#     %1 = tt.splat %0 : (f32) -> tensor<128x128xf32, #blocked>
#     %2 = tt.get_program_id x : i32
#     %3 = arith.muli %2, %c128_i32 : i32
#     %4 = arith.extsi %3 : i32 to i64
#     %5 = tt.splat %4 : (i64) -> tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
#     %6 = tt.splat %4 : (i64) -> tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#     %7 = tt.get_program_id y : i32
#     %8 = arith.muli %7, %arg7 : i32
#     %13 = tt.addptr %arg1, %8 : !tt.ptr<f16, 1>, i32
#     %46 = tt.splat %13 : (!tt.ptr<f16, 1>) -> tensor<128x1x!tt.ptr<f16, 1>, #blocked1>

#     %9 = tt.addptr %arg0, %8 : !tt.ptr<f16, 1>, i32
#     %10 = tt.splat %9 : (!tt.ptr<f16, 1>) -> tensor<128x1x!tt.ptr<f16, 1>, #blocked>
#     %11 = arith.extsi %arg8 : i32 to i64
#     %12 = tt.splat %11 : (i64) -> tensor<128x1xi64, #blocked>

#     %14 = arith.extsi %arg11 : i32 to i64
#     %722 = tt.splat %14 : (i64) -> tensor<1x128xi64, #blocked1>
#     %57 = tt.addptr %arg2, %8 : !tt.ptr<f16, 1>, i32
#     %58 = arith.extsi %arg14 : i32 to i64

#     %15 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
#     %16 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#     %17 = arith.extsi %15 : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> to tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
#     %18 = arith.extsi %16 : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #mfma}>> to tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#     %19 = arith.addi %5, %17 : tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
#     %20 = arith.addi %6, %18 : tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #mfma}>>

#     %21 = tt.expand_dims %19 {axis = 1 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>>) -> tensor<128x1xi64, #blocked>
#     %22 = tt.expand_dims %20 {axis = 1 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #mfma}>>) -> tensor<128x1xi64, #mfma>
#     %23 = arith.muli %21, %12 : tensor<128x1xi64, #blocked>
#     %24 = tt.addptr %10, %23 : tensor<128x1x!tt.ptr<f16, 1>, #blocked>, tensor<128x1xi64, #blocked>
#     %25 = tt.broadcast %24 : (tensor<128x1x!tt.ptr<f16, 1>, #blocked>) -> tensor<128x128x!tt.ptr<f16, 1>, #blocked>
#     %26 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
#     %27 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #mfma}>>

#     %28 = arith.extsi %26 : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> to tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
#     %29 = arith.extsi %27 : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #mfma}>> to tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #mfma}>>
#     %30 = tt.expand_dims %28 {axis = 0 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked}>>) -> tensor<1x128xi64, #blocked>
#     %31 = tt.expand_dims %29 {axis = 0 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #mfma}>>) -> tensor<1x128xi64, #mfma>
#     %32 = tt.broadcast %30 : (tensor<1x128xi64, #blocked>) -> tensor<128x128xi64, #blocked>
#     %33 = tt.broadcast %31 : (tensor<1x128xi64, #mfma>) -> tensor<128x128xi64, #mfma>

#     %34 = tt.addptr %25, %32 : tensor<128x128x!tt.ptr<f16, 1>, #blocked>, tensor<128x128xi64, #blocked>
#     %35 = tt.load %34 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x128xf16, #blocked>
#     %36 = arith.extf %35 : tensor<128x128xf16, #blocked> to tensor<128x128xf32, #blocked>
#     %37 = arith.mulf %36, %1 : tensor<128x128xf32, #blocked>
#     %38 = arith.truncf %37 : tensor<128x128xf32, #blocked> to tensor<128x128xf16, #blocked>

#     %39 = triton_gpu.convert_layout %38 : (tensor<128x128xf16, #blocked>) -> tensor<128x128xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 8}>>
#     %40 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
#     %41 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
#     %43 = arith.extsi %41 : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>> to tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
#     %45 = tt.expand_dims %43 {axis = 1 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>) -> tensor<128x1xi64, #blocked1>
#     %47 = tt.addptr %46, %45 : tensor<128x1x!tt.ptr<f16, 1>, #blocked1>, tensor<128x1xi64, #blocked1>

#     %723 = tt.broadcast %47 : (tensor<128x1x!tt.ptr<f16, 1>, #blocked1>) -> tensor<128x128x!tt.ptr<f16, 1>, #blocked1>
#     %42 = arith.extsi %40 : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>> to tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
#     %44 = tt.expand_dims %42 {axis = 0 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>) -> tensor<1x128xi64, #blocked1>
#     %59 = tt.splat %57 : (!tt.ptr<f16, 1>) -> tensor<128x1x!tt.ptr<f16, 1>, #blocked1>
#     %60 = tt.splat %58 : (i64) -> tensor<1x128xi64, #blocked1>
#     %61 = arith.muli %44, %60 : tensor<1x128xi64, #blocked1>
#     %62 = tt.broadcast %61 : (tensor<1x128xi64, #blocked1>) -> tensor<128x128xi64, #blocked1>
#     %63:5 = scf.for %arg21 = %c0_i32 to %arg20 step %c128_i32 iter_args(%arg23 = %cst_1, %arg24 = %cst, %arg25 = %cst_0, %arg26 = %c0_i64, %arg27 = %c0_i64) -> (tensor<128x128xf32, #mfma>, tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>, tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>, i64, i64)  : i32 {
#       %106 = tt.splat %arg27 : (i64) -> tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
#       %107 = arith.addi %106, %42 : tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
#       %108 = tt.expand_dims %107 {axis = 0 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>) -> tensor<1x128xi64, #blocked1>
#       %109 = arith.muli %108, %722 : tensor<1x128xi64, #blocked1>
#       %110 = tt.broadcast %109 : (tensor<1x128xi64, #blocked1>) -> tensor<128x128xi64, #blocked1>
#       %111 = tt.addptr %723, %110 : tensor<128x128x!tt.ptr<f16, 1>, #blocked1>, tensor<128x128xi64, #blocked1>
#       %112 = tt.load %111 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x128xf16, #blocked1>
#       %113 = triton_gpu.convert_layout %112 : (tensor<128x128xf16, #blocked1>) -> tensor<128x128xf16, #shared>
#       %114 = triton_gpu.convert_layout %113 : (tensor<128x128xf16, #shared>) -> tensor<128x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 8}>>
#       %115 = tt.dot %39, %114, %cst_1 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x128xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 8}>> * tensor<128x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 8}>> -> tensor<128x128xf32, #mfma>
#       %78 = "tt.reduce"(%115) <{axis = 1 : i32}> ({
#       ^bb0(%arg28: f32, %arg29: f32):
#         %206 = arith.maximumf %arg28, %arg29 : f32
#         tt.reduce.return %206 : f32
#       }) : (tensor<128x128xf32, #mfma>) -> tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#       %79 = arith.maximumf %arg25, %78 : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#       %80 = tt.expand_dims %79 {axis = 1 : i32} : (tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>) -> tensor<128x1xf32, #mfma>
#       %81 = tt.broadcast %80 : (tensor<128x1xf32, #mfma>) -> tensor<128x128xf32, #mfma>
#       %82 = arith.subf %115, %81 : tensor<128x128xf32, #mfma>

#       %83 = tt.extern_elementwise %82 {libname = "libdevice", libpath = "/triton/python/triton/language/../third_party/hip/lib/bitcode/cuda2gcn.bc", pure = true, symbol = "__nv_exp2f"} : (tensor<128x128xf32, #mfma>) -> tensor<128x128xf32, #mfma>
#       %96 = arith.truncf %83 : tensor<128x128xf32, #mfma> to tensor<128x128xf16, #mfma>
#       %97 = triton_gpu.convert_layout %96 : (tensor<128x128xf16, #mfma>) -> tensor<128x128xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>

#       %84 = arith.subf %arg25, %79 : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#       %85 = tt.extern_elementwise %84 {libname = "libdevice", libpath = "/triton/python/triton/language/../third_party/hip/lib/bitcode/cuda2gcn.bc", pure = true, symbol = "__nv_exp2f"} : (tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>) -> tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#       %86 = tt.expand_dims %85 {axis = 1 : i32} : (tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>) -> tensor<128x1xf32, #mfma>
#       %87 = tt.broadcast %86 : (tensor<128x1xf32, #mfma>) -> tensor<128x128xf32, #mfma>
#       %88 = arith.mulf %arg23, %87 : tensor<128x128xf32, #mfma>
#       %89 = tt.splat %arg26 : (i64) -> tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
#       %90 = arith.addi %89, %43 : tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
#       %91 = tt.expand_dims %90 {axis = 1 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>) -> tensor<128x1xi64, #blocked1>
#       %92 = tt.addptr %59, %91 : tensor<128x1x!tt.ptr<f16, 1>, #blocked1>, tensor<128x1xi64, #blocked1>
#       %93 = tt.broadcast %92 : (tensor<128x1x!tt.ptr<f16, 1>, #blocked1>) -> tensor<128x128x!tt.ptr<f16, 1>, #blocked1>
#       %94 = tt.addptr %93, %62 : tensor<128x128x!tt.ptr<f16, 1>, #blocked1>, tensor<128x128xi64, #blocked1>
#       %95 = tt.load %94 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x128xf16, #blocked1>
#       %98 = triton_gpu.convert_layout %95 : (tensor<128x128xf16, #blocked1>) -> tensor<128x128xf16, #shared1>
#       %99 = triton_gpu.convert_layout %98 : (tensor<128x128xf16, #shared1>) -> tensor<128x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>>
#       %100 = tt.dot %97, %99, %88 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x128xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>> * tensor<128x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>> -> tensor<128x128xf32, #mfma>
#       %101 = "tt.reduce"(%83) <{axis = 1 : i32}> ({
#       ^bb0(%arg28: f32, %arg29: f32):
#         %206 = arith.addf %arg28, %arg29 : f32
#         tt.reduce.return %206 : f32
#       }) : (tensor<128x128xf32, #mfma>) -> tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#       %102 = arith.mulf %arg24, %85 : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#       %103 = arith.addf %102, %101 : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#       %104 = arith.addi %arg26, %c128_i64 : i64
#       %105 = arith.addi %arg27, %c128_i64 : i64
#       scf.yield  %100, %103, %79, %104, %105 : tensor<128x128xf32, #mfma>, tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>, tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>, i64, i64
#     }
#     %64 = tt.expand_dims %63#1 {axis = 1 : i32} : (tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>) -> tensor<128x1xf32, #mfma>
#     %65 = tt.broadcast %64 : (tensor<128x1xf32, #mfma>) -> tensor<128x128xf32, #mfma>
#     %66 = arith.divf %63#0, %65 : tensor<128x128xf32, #mfma>
#     %67 = tt.addptr %arg5, %8 : !tt.ptr<f16, 1>, i32
#     %68 = arith.extsi %arg17 : i32 to i64
#     %69 = arith.truncf %66 : tensor<128x128xf32, #mfma> to tensor<128x128xf16, #mfma>
#     %70 = tt.splat %68 : (i64) -> tensor<128x1xi64, #mfma>
#     %71 = arith.muli %22, %70 : tensor<128x1xi64, #mfma>
#     %72 = tt.splat %67 : (!tt.ptr<f16, 1>) -> tensor<128x1x!tt.ptr<f16, 1>, #mfma>
#     %73 = tt.addptr %72, %71 : tensor<128x1x!tt.ptr<f16, 1>, #mfma>, tensor<128x1xi64, #mfma>
#     %74 = tt.broadcast %73 : (tensor<128x1x!tt.ptr<f16, 1>, #mfma>) -> tensor<128x128x!tt.ptr<f16, 1>, #mfma>
#     %75 = tt.addptr %74, %33 : tensor<128x128x!tt.ptr<f16, 1>, #mfma>, tensor<128x128xi64, #mfma>
#     tt.store %75, %69 {cache = 1 : i32, evict = 1 : i32} : tensor<128x128xf16, #mfma>
#     tt.return
#   }
# }

# """


# ir = """
# #blocked = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [32, 2], warpsPerCTA = [4, 1], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
# #blocked1 = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [4, 16], warpsPerCTA = [1, 4], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
# #mfma = #triton_gpu.mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>
# #shared = #triton_gpu.shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = false}>
# #shared1 = #triton_gpu.shared<{vec = 4, perPhase = 2, maxPhase = 8, order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = false}>
# module attributes {"triton_gpu.compute-capability" = 0 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {
#   tt.func public @_attn_fwd_0d1d2d34d5d6de7de8de9c10de11de12de13c14de15de16de17c18de19de20de21c2223de24de(%arg0: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg3: f32, %arg4: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg7: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg8: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg9: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg10: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg11: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg12: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg13: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg14: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg15: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg16: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg17: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg18: i32, %arg19: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg20: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}) attributes {noinline = false} {
#     %cst = arith.constant dense<1.000000e+00> : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#     %cst_0 = arith.constant dense<0xFF800000> : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#     %cst_1 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mfma>
#     %c128_i64 = arith.constant 128 : i64
#     %c0_i64 = arith.constant 0 : i64
#     %c128_i32 = arith.constant 128 : i32
#     %c0_i32 = arith.constant 0 : i32
#     %cst_2 = arith.constant 1.44269502 : f32
#     %0 = arith.mulf %arg3, %cst_2 : f32
#     %1 = tt.splat %0 : (f32) -> tensor<128x128xf32, #blocked>
#     %2 = tt.get_program_id x : i32
#     %3 = arith.muli %2, %c128_i32 : i32
#     %4 = arith.extsi %3 : i32 to i64
#     %5 = tt.splat %4 : (i64) -> tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
#     %6 = tt.splat %4 : (i64) -> tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#     %7 = tt.get_program_id y : i32
#     %8 = arith.muli %7, %arg7 : i32
#     %9 = tt.addptr %arg1, %8 : !tt.ptr<f16, 1>, i32
#     %10 = tt.splat %9 : (!tt.ptr<f16, 1>) -> tensor<128x1x!tt.ptr<f16, 1>, #blocked1>
#     %11 = tt.addptr %arg0, %8 : !tt.ptr<f16, 1>, i32
#     %12 = tt.splat %11 : (!tt.ptr<f16, 1>) -> tensor<128x1x!tt.ptr<f16, 1>, #blocked>
#     %13 = arith.extsi %arg8 : i32 to i64
#     %14 = tt.splat %13 : (i64) -> tensor<128x1xi64, #blocked>
#     %15 = arith.extsi %arg11 : i32 to i64
#     %16 = tt.splat %15 : (i64) -> tensor<1x128xi64, #blocked1>
#     %17 = tt.addptr %arg2, %8 : !tt.ptr<f16, 1>, i32
#     %18 = arith.extsi %arg14 : i32 to i64
#     %19 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
#     %20 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#     %21 = arith.extsi %19 : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> to tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
#     %22 = arith.extsi %20 : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #mfma}>> to tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#     %23 = arith.addi %5, %21 : tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
#     %24 = arith.addi %6, %22 : tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#     %25 = tt.expand_dims %23 {axis = 1 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>>) -> tensor<128x1xi64, #blocked>
#     %26 = tt.expand_dims %24 {axis = 1 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #mfma}>>) -> tensor<128x1xi64, #mfma>
#     %27 = arith.muli %25, %14 : tensor<128x1xi64, #blocked>
#     %28 = tt.addptr %12, %27 : tensor<128x1x!tt.ptr<f16, 1>, #blocked>, tensor<128x1xi64, #blocked>
#     %29 = tt.broadcast %28 : (tensor<128x1x!tt.ptr<f16, 1>, #blocked>) -> tensor<128x128x!tt.ptr<f16, 1>, #blocked>
#     %30 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
#     %31 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #mfma}>>
#     %32 = arith.extsi %30 : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> to tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
#     %33 = arith.extsi %31 : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #mfma}>> to tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #mfma}>>
#     %34 = tt.expand_dims %32 {axis = 0 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked}>>) -> tensor<1x128xi64, #blocked>
#     %35 = tt.expand_dims %33 {axis = 0 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #mfma}>>) -> tensor<1x128xi64, #mfma>
#     %36 = tt.broadcast %34 : (tensor<1x128xi64, #blocked>) -> tensor<128x128xi64, #blocked>
#     %37 = tt.broadcast %35 : (tensor<1x128xi64, #mfma>) -> tensor<128x128xi64, #mfma>
#     %38 = tt.addptr %29, %36 : tensor<128x128x!tt.ptr<f16, 1>, #blocked>, tensor<128x128xi64, #blocked>
#     %39 = triton_gpu.view_slice %38[0, 96] [128, 32] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked> to tensor<128x32x!tt.ptr<f16, 1>, #blocked>
#     %40 = triton_gpu.view_slice %38[0, 64] [128, 32] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked> to tensor<128x32x!tt.ptr<f16, 1>, #blocked>
#     %41 = triton_gpu.view_slice %38[0, 32] [128, 32] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked> to tensor<128x32x!tt.ptr<f16, 1>, #blocked>
#     %42 = triton_gpu.view_slice %38[0, 0] [128, 32] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked> to tensor<128x32x!tt.ptr<f16, 1>, #blocked>
#     %43 = tt.load %39 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x32xf16, #blocked>
#     %44 = tt.load %40 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x32xf16, #blocked>
#     %45 = tt.load %41 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x32xf16, #blocked>
#     %46 = tt.load %42 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x32xf16, #blocked>
#     %47 = arith.extf %43 : tensor<128x32xf16, #blocked> to tensor<128x32xf32, #blocked>
#     %48 = arith.extf %44 : tensor<128x32xf16, #blocked> to tensor<128x32xf32, #blocked>
#     %49 = arith.extf %45 : tensor<128x32xf16, #blocked> to tensor<128x32xf32, #blocked>
#     %50 = arith.extf %46 : tensor<128x32xf16, #blocked> to tensor<128x32xf32, #blocked>
#     %51 = triton_gpu.view_slice %1[0, 96] [128, 32] [1, 1] : tensor<128x128xf32, #blocked> to tensor<128x32xf32, #blocked>
#     %52 = triton_gpu.view_slice %1[0, 64] [128, 32] [1, 1] : tensor<128x128xf32, #blocked> to tensor<128x32xf32, #blocked>
#     %53 = triton_gpu.view_slice %1[0, 32] [128, 32] [1, 1] : tensor<128x128xf32, #blocked> to tensor<128x32xf32, #blocked>
#     %54 = triton_gpu.view_slice %1[0, 0] [128, 32] [1, 1] : tensor<128x128xf32, #blocked> to tensor<128x32xf32, #blocked>
#     %55 = arith.mulf %47, %51 : tensor<128x32xf32, #blocked>
#     %56 = arith.mulf %48, %52 : tensor<128x32xf32, #blocked>
#     %57 = arith.mulf %49, %53 : tensor<128x32xf32, #blocked>
#     %58 = arith.mulf %50, %54 : tensor<128x32xf32, #blocked>
#     %59 = arith.truncf %55 : tensor<128x32xf32, #blocked> to tensor<128x32xf16, #blocked>
#     %60 = arith.truncf %56 : tensor<128x32xf32, #blocked> to tensor<128x32xf16, #blocked>
#     %61 = arith.truncf %57 : tensor<128x32xf32, #blocked> to tensor<128x32xf16, #blocked>
#     %62 = arith.truncf %58 : tensor<128x32xf32, #blocked> to tensor<128x32xf16, #blocked>
#     %63 = triton_gpu.convert_layout %59 : (tensor<128x32xf16, #blocked>) -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 8}>>
#     %64 = triton_gpu.convert_layout %60 : (tensor<128x32xf16, #blocked>) -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 8}>>
#     %65 = triton_gpu.convert_layout %61 : (tensor<128x32xf16, #blocked>) -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 8}>>
#     %66 = triton_gpu.convert_layout %62 : (tensor<128x32xf16, #blocked>) -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 8}>>
#     %67 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
#     %68 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
#     %69 = arith.extsi %68 : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>> to tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
#     %70 = tt.expand_dims %69 {axis = 1 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>) -> tensor<128x1xi64, #blocked1>
#     %71 = tt.addptr %10, %70 : tensor<128x1x!tt.ptr<f16, 1>, #blocked1>, tensor<128x1xi64, #blocked1>
#     %72 = tt.broadcast %71 : (tensor<128x1x!tt.ptr<f16, 1>, #blocked1>) -> tensor<128x128x!tt.ptr<f16, 1>, #blocked1>
#     %73 = arith.extsi %67 : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>> to tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
#     %74 = tt.expand_dims %73 {axis = 0 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>) -> tensor<1x128xi64, #blocked1>
#     %75 = tt.splat %17 : (!tt.ptr<f16, 1>) -> tensor<128x1x!tt.ptr<f16, 1>, #blocked1>
#     %76 = tt.splat %18 : (i64) -> tensor<1x128xi64, #blocked1>
#     %77 = arith.muli %74, %76 : tensor<1x128xi64, #blocked1>
#     %78 = tt.broadcast %77 : (tensor<1x128xi64, #blocked1>) -> tensor<128x128xi64, #blocked1>
#     %79:5 = scf.for %arg21 = %c0_i32 to %arg20 step %c128_i32 iter_args(%arg22 = %cst_1, %arg23 = %cst, %arg24 = %cst_0, %arg25 = %c0_i64, %arg26 = %c0_i64) -> (tensor<128x128xf32, #mfma>, tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>, tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>, i64, i64)  : i32 {
#       %92 = tt.splat %arg26 : (i64) -> tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
#       %93 = arith.addi %92, %73 : tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
#       %94 = tt.expand_dims %93 {axis = 0 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>) -> tensor<1x128xi64, #blocked1>
#       %95 = arith.muli %94, %16 : tensor<1x128xi64, #blocked1>
#       %96 = tt.broadcast %95 : (tensor<1x128xi64, #blocked1>) -> tensor<128x128xi64, #blocked1>
#       %97 = tt.addptr %72, %96 : tensor<128x128x!tt.ptr<f16, 1>, #blocked1>, tensor<128x128xi64, #blocked1>
#       %98 = triton_gpu.view_slice %97[96, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked1> to tensor<32x128x!tt.ptr<f16, 1>, #blocked1>
#       %99 = triton_gpu.view_slice %97[64, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked1> to tensor<32x128x!tt.ptr<f16, 1>, #blocked1>
#       %100 = triton_gpu.view_slice %97[32, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked1> to tensor<32x128x!tt.ptr<f16, 1>, #blocked1>
#       %101 = triton_gpu.view_slice %97[0, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked1> to tensor<32x128x!tt.ptr<f16, 1>, #blocked1>
#       %102 = tt.load %101 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #blocked1>
#       gpu.barrier
#       %103 = triton_gpu.convert_layout %102 : (tensor<32x128xf16, #blocked1>) -> tensor<32x128xf16, #shared>
#       %104 = tt.load %100 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #blocked1>
#       gpu.barrier
#       %105 = triton_gpu.convert_layout %103 : (tensor<32x128xf16, #shared>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 8}>>
#       %106 = tt.dot %66, %105, %cst_1 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 8}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 8}>> -> tensor<128x128xf32, #mfma>
#       gpu.barrier
#       %107 = triton_gpu.convert_layout %104 : (tensor<32x128xf16, #blocked1>) -> tensor<32x128xf16, #shared>
#       %108 = tt.load %99 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #blocked1>
#       gpu.barrier
#       %109 = triton_gpu.convert_layout %107 : (tensor<32x128xf16, #shared>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 8}>>
#       %110 = tt.dot %65, %109, %106 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 8}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 8}>> -> tensor<128x128xf32, #mfma>
#       gpu.barrier
#       %111 = triton_gpu.convert_layout %108 : (tensor<32x128xf16, #blocked1>) -> tensor<32x128xf16, #shared>
#       %112 = tt.load %98 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #blocked1>
#       gpu.barrier
#       %113 = triton_gpu.convert_layout %111 : (tensor<32x128xf16, #shared>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 8}>>
#       %114 = tt.dot %64, %113, %110 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 8}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 8}>> -> tensor<128x128xf32, #mfma>
#       gpu.barrier
#       %115 = triton_gpu.convert_layout %112 : (tensor<32x128xf16, #blocked1>) -> tensor<32x128xf16, #shared>
#       gpu.barrier
#       %116 = triton_gpu.convert_layout %115 : (tensor<32x128xf16, #shared>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 8}>>
#       %117 = tt.dot %63, %116, %114 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 8}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 8}>> -> tensor<128x128xf32, #mfma>
#       %118 = "tt.reduce"(%117) <{axis = 1 : i32}> ({
#       ^bb0(%arg27: f32, %arg28: f32):
#         %169 = arith.maximumf %arg27, %arg28 : f32
#         tt.reduce.return %169 : f32
#       }) : (tensor<128x128xf32, #mfma>) -> tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#       %119 = arith.maximumf %arg24, %118 : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#       %120 = tt.expand_dims %119 {axis = 1 : i32} : (tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>) -> tensor<128x1xf32, #mfma>
#       %121 = tt.broadcast %120 : (tensor<128x1xf32, #mfma>) -> tensor<128x128xf32, #mfma>
#       %122 = arith.subf %117, %121 : tensor<128x128xf32, #mfma>
#       %123 = tt.extern_elementwise %122 {libname = "libdevice", libpath = "/triton/python/triton/language/../third_party/hip/lib/bitcode/cuda2gcn.bc", pure = true, symbol = "__nv_exp2f"} : (tensor<128x128xf32, #mfma>) -> tensor<128x128xf32, #mfma>
#       %124 = arith.truncf %123 : tensor<128x128xf32, #mfma> to tensor<128x128xf16, #mfma>
#       %125 = triton_gpu.view_slice %124[0, 96] [128, 32] [1, 1] : tensor<128x128xf16, #mfma> to tensor<128x32xf16, #mfma>
#       %126 = triton_gpu.view_slice %124[0, 64] [128, 32] [1, 1] : tensor<128x128xf16, #mfma> to tensor<128x32xf16, #mfma>
#       %127 = triton_gpu.view_slice %124[0, 32] [128, 32] [1, 1] : tensor<128x128xf16, #mfma> to tensor<128x32xf16, #mfma>
#       %128 = triton_gpu.view_slice %124[0, 0] [128, 32] [1, 1] : tensor<128x128xf16, #mfma> to tensor<128x32xf16, #mfma>
#       %129 = triton_gpu.convert_layout %125 : (tensor<128x32xf16, #mfma>) -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>
#       %130 = triton_gpu.convert_layout %126 : (tensor<128x32xf16, #mfma>) -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>
#       %131 = triton_gpu.convert_layout %127 : (tensor<128x32xf16, #mfma>) -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>
#       %132 = triton_gpu.convert_layout %128 : (tensor<128x32xf16, #mfma>) -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>
#       %133 = arith.subf %arg24, %119 : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#       %134 = tt.extern_elementwise %133 {libname = "libdevice", libpath = "/triton/python/triton/language/../third_party/hip/lib/bitcode/cuda2gcn.bc", pure = true, symbol = "__nv_exp2f"} : (tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>) -> tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#       %135 = tt.expand_dims %134 {axis = 1 : i32} : (tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>) -> tensor<128x1xf32, #mfma>
#       %136 = tt.broadcast %135 : (tensor<128x1xf32, #mfma>) -> tensor<128x128xf32, #mfma>
#       %137 = arith.mulf %arg22, %136 : tensor<128x128xf32, #mfma>
#       %138 = tt.splat %arg25 : (i64) -> tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
#       %139 = arith.addi %138, %69 : tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
#       %140 = tt.expand_dims %139 {axis = 1 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>) -> tensor<128x1xi64, #blocked1>
#       %141 = tt.addptr %75, %140 : tensor<128x1x!tt.ptr<f16, 1>, #blocked1>, tensor<128x1xi64, #blocked1>
#       %142 = tt.broadcast %141 : (tensor<128x1x!tt.ptr<f16, 1>, #blocked1>) -> tensor<128x128x!tt.ptr<f16, 1>, #blocked1>
#       %143 = tt.addptr %142, %78 : tensor<128x128x!tt.ptr<f16, 1>, #blocked1>, tensor<128x128xi64, #blocked1>
#       %144 = triton_gpu.view_slice %143[96, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked1> to tensor<32x128x!tt.ptr<f16, 1>, #blocked1>
#       %145 = triton_gpu.view_slice %143[64, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked1> to tensor<32x128x!tt.ptr<f16, 1>, #blocked1>
#       %146 = triton_gpu.view_slice %143[32, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked1> to tensor<32x128x!tt.ptr<f16, 1>, #blocked1>
#       %147 = triton_gpu.view_slice %143[0, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked1> to tensor<32x128x!tt.ptr<f16, 1>, #blocked1>
#       %148 = tt.load %147 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #blocked1>
#       %150 = tt.load %146 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #blocked1>
#       %154 = tt.load %145 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #blocked1>
#       %158 = tt.load %144 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #blocked1>
#       gpu.barrier
#       %149 = triton_gpu.convert_layout %148 : (tensor<32x128xf16, #blocked1>) -> tensor<32x128xf16, #shared1>
#       %153 = triton_gpu.convert_layout %150 : (tensor<32x128xf16, #blocked1>) -> tensor<32x128xf16, #shared1>
#       %157 = triton_gpu.convert_layout %154 : (tensor<32x128xf16, #blocked1>) -> tensor<32x128xf16, #shared1>
#       %161 = triton_gpu.convert_layout %158 : (tensor<32x128xf16, #blocked1>) -> tensor<32x128xf16, #shared1>
#       gpu.barrier
#       %151 = triton_gpu.convert_layout %149 : (tensor<32x128xf16, #shared1>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>>
#       %155 = triton_gpu.convert_layout %153 : (tensor<32x128xf16, #shared1>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>>
#       %159 = triton_gpu.convert_layout %157 : (tensor<32x128xf16, #shared1>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>>
#       %162 = triton_gpu.convert_layout %161 : (tensor<32x128xf16, #shared1>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>>

#       %152 = tt.dot %132, %151, %137 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>> -> tensor<128x128xf32, #mfma>
#       %156 = tt.dot %131, %155, %152 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>> -> tensor<128x128xf32, #mfma>
#       %160 = tt.dot %130, %159, %156 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>> -> tensor<128x128xf32, #mfma>
#       %163 = tt.dot %129, %162, %160 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>> -> tensor<128x128xf32, #mfma>
#       %164 = "tt.reduce"(%123) <{axis = 1 : i32}> ({
#       ^bb0(%arg27: f32, %arg28: f32):
#         %169 = arith.addf %arg27, %arg28 : f32
#         tt.reduce.return %169 : f32
#       }) : (tensor<128x128xf32, #mfma>) -> tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#       %165 = arith.mulf %arg23, %134 : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#       %166 = arith.addf %165, %164 : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#       %167 = arith.addi %arg25, %c128_i64 : i64
#       %168 = arith.addi %arg26, %c128_i64 : i64
#       scf.yield %163, %166, %119, %167, %168 : tensor<128x128xf32, #mfma>, tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>, tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>, i64, i64
#     }
#     %80 = tt.expand_dims %79#1 {axis = 1 : i32} : (tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>) -> tensor<128x1xf32, #mfma>
#     %81 = tt.broadcast %80 : (tensor<128x1xf32, #mfma>) -> tensor<128x128xf32, #mfma>
#     %82 = arith.divf %79#0, %81 : tensor<128x128xf32, #mfma>
#     %83 = tt.addptr %arg5, %8 : !tt.ptr<f16, 1>, i32
#     %84 = arith.extsi %arg17 : i32 to i64
#     %85 = arith.truncf %82 : tensor<128x128xf32, #mfma> to tensor<128x128xf16, #mfma>
#     %86 = tt.splat %84 : (i64) -> tensor<128x1xi64, #mfma>
#     %87 = arith.muli %26, %86 : tensor<128x1xi64, #mfma>
#     %88 = tt.splat %83 : (!tt.ptr<f16, 1>) -> tensor<128x1x!tt.ptr<f16, 1>, #mfma>
#     %89 = tt.addptr %88, %87 : tensor<128x1x!tt.ptr<f16, 1>, #mfma>, tensor<128x1xi64, #mfma>
#     %90 = tt.broadcast %89 : (tensor<128x1x!tt.ptr<f16, 1>, #mfma>) -> tensor<128x128x!tt.ptr<f16, 1>, #mfma>
#     %91 = tt.addptr %90, %37 : tensor<128x128x!tt.ptr<f16, 1>, #mfma>, tensor<128x128xi64, #mfma>
#     tt.store %91, %85 {cache = 1 : i32, evict = 1 : i32} : tensor<128x128xf16, #mfma>
#     tt.return
#   }
# }

# """
# Pick the fp8 data type

# AMD E5M2B16
# float8:tl.constexpr = torch.float8_e5m2fnuz

# AMD E4M3B8
# Note: When picking this f8 data type, scaling is required when using f8
# for the second gemm
TORCH_HAS_FP8E4 = hasattr(torch, 'float8_e4m3fnuz')
float8:tl.constexpr = None if not TORCH_HAS_FP8E4 else torch.float8_e4m3fnuz

@triton.jit
def max_fn(x, y):
    return tl.math.max(x, y)


import tempfile
with tempfile.NamedTemporaryFile(mode='w', suffix='.ttgir') as f:
    f.write(ir)
    f.flush()
    kernel = triton.compile(f.name)

@triton.jit
def _attn_fwd(
    Q, K, V, sm_scale, M, Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, H,
    N_CTX,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    pre_load_v: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    qkv_offset = off_hz * stride_qh
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qkv_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + qkv_offset,
        shape=(BLOCK_DMODEL, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1)
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + qkv_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(0, 1)
    )
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # scale sm_scale by log_2(e) and use
    # 2^x instead of exp in the loop because CSE and LICM
    # don't work as expected with `exp` in the loop
    qk_scale = sm_scale * 1.44269504
    q = tl.load(Q_block_ptr)
    # it's even better to multiply the qk_scale and convert to f16
    # than doing it inside the loop
    # So conversion is quite cheap
    q = (q * qk_scale).to(q.dtype)
    lo, hi = 0, N_CTX
    # loop over k, v and update accumulator
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = tl.load(K_block_ptr)
        if pre_load_v:
            v = tl.load(V_block_ptr)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        #qk = (qk * qk_scale)
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk = qk - m_ij[:, None]
        p = tl.math.exp2(qk)
        # -- update output accumulator --
        alpha = tl.math.exp2(m_i - m_ij)
        acc = acc * alpha[:, None]
        if not pre_load_v:
            v = tl.load(V_block_ptr)
        acc += tl.dot(p.to(v.dtype), v)
        # -- update m_i and l_i
        l_ij = tl.sum(p, 1)
        l_i = l_i * alpha + l_ij
        # update m_i and l_i
        m_i = m_ij
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
    acc = acc / l_i[:, None]
    # write back O
    O_block_ptr = tl.make_block_ptr(
        base=Out + qkv_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    tl.store(O_block_ptr, acc.to(Out.type.element_ty))


empty = torch.empty(128, device="cuda")


class _attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, sm_scale):
        # shape constraints
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-2]
        assert Lq == Lk and Lk == Lv
        assert Lk in {16, 32, 64, 128}
        o = torch.empty_like(q, dtype=v.dtype)
        if torch.version.hip is None:
            BLOCK_M = 128
            BLOCK_N = 64 if Lk <= 64 else 32
            num_stages = 4 if Lk <= 64 else 3
            num_warps = 4 if Lk <= 64 else 8

        ## hardcoded best perf_configs for MI250
        if Lk == 64:
            ## D_HEAD = 64
            BLOCK_M = 128
            BLOCK_N = 64
            waves_per_eu = 3
            num_warps = 4
            num_stages = 1
            ## causal=False likes to pre load v but causal=True does not
            pre_load_v = False if causal else True
            slice_k_tile = 32
            kpack = 1
        else:
            ## D_HEAD = 128
            ## For fp16, pick BLOCK_M=256, num_warps=8
            ## For fp8, pick BLOCK_M=128, num_warps=4
            ## TODO (zhanglx): add tuning infra for FA
            BLOCK_M = 128 #if TORCH_HAS_FP8E4 and q.dtype == torch.float8_e4m3fnuz else 256
            BLOCK_N = 128
            waves_per_eu = 2
            num_warps = 4
            num_stages = 1
            pre_load_v = False
            slice_k_tile = 32
            kpack = 2

        grid = ( triton.cdiv(q.shape[2], BLOCK_M), q.shape[0] * q.shape[1], 1)
        M = torch.empty((q.shape[0] * q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)




        kernel[(32, 192, 1)](
            q, k, v, sm_scale, M, o,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2), 
            v.stride(0), v.stride(1), v.stride(2),
            o.stride(0), o.stride(1), o.stride(2),
            q.shape[0], q.shape[1],
            q.shape[2],
            # BLOCK_DMODEL=Lk,
            # BLOCK_M = BLOCK_M,
            # BLOCK_N = BLOCK_N,
            # waves_per_eu = waves_per_eu,
            # num_warps = num_warps,
            # num_stages = num_stages,
            # pre_load_v = pre_load_v,
            # slice_k_tile = slice_k_tile,
            # kpack = kpack,
        )


        return o


attention = _attention.apply

name_to_torch_types = {
    'fp16': torch.float16,
    'bf16': torch.bfloat16,
    'fp8': float8
}

@pytest.mark.parametrize('Z, H, N_CTX, D_HEAD, dtype',
[ (*shape, dtype)
    for shape in [(4, 48, 4096, 128)]
    for dtype in ['fp16']])
def test_op_fwd(Z, H, N_CTX, D_HEAD, dtype):
    torch.manual_seed(20)
    init_dtype = torch.float16 if dtype == 'fp8' else name_to_torch_types[dtype]
    q = (
        torch.empty((Z, H, N_CTX, D_HEAD), dtype=init_dtype, device="cuda")
        .normal_(mean=0., std=0.5)
        .requires_grad_()
    )
    k = (
        torch.empty((Z, H, N_CTX, D_HEAD), dtype=init_dtype, device="cuda")
        .normal_(mean=0., std=0.5)
        .requires_grad_()
    )
    v = (
        torch.empty((Z, H, D_HEAD, N_CTX), dtype=init_dtype, device="cuda")
        .normal_(mean=0., std=0.5)
        .requires_grad_()
    )
    sm_scale = 0.5
    # reference implementation
    M = torch.tril(torch.ones((N_CTX, N_CTX), device="cuda"))
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    p = torch.softmax(p.float(), dim=-1).to(q.dtype)
    ref_out = torch.matmul(p, v.transpose(2,3))
    # triton implementation
    # q,k casting for partial fp8
    q = q.to(name_to_torch_types[dtype])
    k = k.to(name_to_torch_types[dtype])
    dout = torch.randn_like(q, dtype=torch.float16)
    tri_out = attention(q, k, v, sm_scale)
    # compare
    atol = 1.4e-1 if dtype == 'fp8' else 1e-2
    rtol = 1e-2 if dtype == 'fp8' else 3e-3
    torch.testing.assert_close(ref_out, tri_out, atol=atol, rtol=rtol)


try:
    from flash_attn.flash_attn_interface import \
        flash_attn_qkvpacked_func as flash_attn_func
    FLASH_VER = 2
except BaseException:
    try:
        from flash_attn.flash_attn_interface import flash_attn_func
        FLASH_VER = 1
    except BaseException:
        FLASH_VER = None
HAS_FLASH = FLASH_VER is not None

# vary seq length for fixed head and batch=4
configs = []
for dtype in ['fp16']:
    for D_HEAD in [128]:
        for causal in [False]:
            configs.append(triton.testing.Benchmark(
                x_names=['BATCH', 'H','N_CTX'],
                x_vals=[#(16, 16, 1024),
                        # (8, 16, 2048),
                        # (4, 16, 4096),
                        # (2, 16, 8192),
                        # (1, 16, 16384),
                        # (4, 48, 1024),
                        # (4, 48, 2048),
                        (4, 48, 4096),
                        # (4, 48, 8192),
                        # (4, 48, 16384),
                        ],
                line_arg='provider',
                line_vals=['triton'],
                line_names=['Triton'],
                #styles=[('red', '-'), ('blue', '-')],
                ylabel='ms',
                plot_name=f'fused-attention-fwd-d{D_HEAD}-causal={causal}-{dtype}',
                args={
                    'D_HEAD': D_HEAD,
                    'dtype': dtype,
                    'causal': causal})
            )


@triton.testing.perf_report(configs)
def bench_flash_attention(BATCH, H, N_CTX, D_HEAD, causal, provider, dtype, device="cuda"):
    if dtype == 'fp8' and not TORCH_HAS_FP8E4:
        sys.exit("fp8 is not available")
    warmup = 25
    rep = 100
    init_dtype = torch.float16 if dtype != 'bf16' else torch.bfloat16
    q = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=init_dtype, device="cuda", requires_grad=True)
    k = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=init_dtype, device="cuda", requires_grad=True)
    v = torch.randn((BATCH, H, D_HEAD, N_CTX), dtype=init_dtype, device="cuda", requires_grad=True)
    sm_scale = 1.3
    # q,k casting for partial fp8
    q = q.to(name_to_torch_types[dtype])
    k = k.to(name_to_torch_types[dtype])
    fn = lambda: attention(q, k, v, sm_scale)
    ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    flops_per_matmul = 2. * BATCH * H * N_CTX * N_CTX * D_HEAD
    total_flops = 2 * flops_per_matmul
    return total_flops / ms * 1e-9


def main():
    bench_flash_attention.run(save_path='.', print_data=True)

if __name__ == '__main__':
    sys.exit(main())
