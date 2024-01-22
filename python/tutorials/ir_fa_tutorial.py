"""
Fused Attention
===============

This is a Triton implementation of the Flash Attention v2 algorithm from Tri Dao (https://tridao.me/publications/flash2/flash2.pdf)
Credits: OpenAI kernel team

Extra Credits:
- Original flash attention paper (https://arxiv.org/abs/2205.14135)
- Rabe and Staats (https://arxiv.org/pdf/2112.05682v2.pdf)

"""

import pytest
import torch

import triton
import triton.language as tl

torch_dtype:tl.constexpr = torch.float16
TORCH_HAS_FP8 = False
TORCH_HAS_FP8E5 = hasattr(torch, 'float8_e5m2')
TORCH_HAS_FP8E5FNUZ = hasattr(torch, 'float8_e5m2fnuz')
if TORCH_HAS_FP8E5:
    torch_dtype:tl.constexpr = torch.float8_e5m2
    TORCH_HAS_FP8 = True
if TORCH_HAS_FP8E5FNUZ:
    torch_dtype:tl.constexpr = torch.float8_e5m2fnuz
    TORCH_HAS_FP8 = True


empty = torch.empty(128, device="cuda")
# ir = """
# #blocked = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
# #blocked1 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
# #blocked2 = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 4], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
# #blocked3 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
# #mfma = #triton_gpu.mfma<{nonKDim = 32, warpsPerCTA = [4, 1], isTransposed = true}>
# #shared = #triton_gpu.shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = false}>
# #shared1 = #triton_gpu.shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = false}>
# #shared2 = #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = false}>
# module attributes {"triton_gpu.compute-capability" = 0 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {
#   tt.func public @_attn_fwd_0d1d2d34d5d6de7de8de9c10de11de12de13c14de15de16de17c18de19de20de21c2223de24de(%arg0: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg3: f32, %arg4: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg7: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg8: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg9: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg10: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg11: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg12: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg13: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg14: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg15: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg16: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg17: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg18: i32, %arg19: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg20: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}) attributes {noinline = false} {
#     %c128_i32 = arith.constant 128 : i32
#     %c0_i32 = arith.constant 0 : i32
#     %cst = arith.constant dense<1.000000e+00> : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#     %cst_0 = arith.constant dense<0xFF800000> : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#     %cst_1 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mfma>
#     %c0_i64 = arith.constant 0 : i64
#     %c128_i64 = arith.constant 128 : i64
#     %cst_2 = arith.constant 1.44269502 : f32
#     %0 = arith.mulf %arg3, %cst_2 : f32
#     %1 = tt.splat %0 : (f32) -> tensor<128x128xf32, #blocked>
#     %2 = triton_gpu.view_slice %1[0, 64] [128, 64] [1, 1] : tensor<128x128xf32, #blocked> to tensor<128x64xf32, #blocked>
#     %3 = triton_gpu.view_slice %1[0, 0] [128, 64] [1, 1] : tensor<128x128xf32, #blocked> to tensor<128x64xf32, #blocked>
#     %4 = tt.get_program_id x : i32
#     %5 = arith.muli %4, %c128_i32 : i32
#     %6 = tt.splat %5 : (i32) -> tensor<128xi32, #blocked1>
#     %7 = arith.extsi %5 : i32 to i64
#     %8 = tt.splat %7 : (i64) -> tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#     %9 = tt.splat %7 : (i64) -> tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
#     %10 = tt.get_program_id y : i32
#     %11 = arith.muli %10, %arg20 : i32
#     %12 = tt.addptr %arg4, %11 : !tt.ptr<f32, 1>, i32
#     %13 = tt.splat %12 : (!tt.ptr<f32, 1>) -> tensor<128x!tt.ptr<f32, 1>, #blocked1>
#     %14 = arith.muli %10, %arg7 : i32
#     %15 = tt.addptr %arg5, %14 : !tt.ptr<f16, 1>, i32
#     %16 = tt.splat %15 : (!tt.ptr<f16, 1>) -> tensor<128x1x!tt.ptr<f16, 1>, #mfma>
#     %17 = tt.addptr %arg1, %14 : !tt.ptr<f16, 1>, i32
#     %18 = tt.splat %17 : (!tt.ptr<f16, 1>) -> tensor<128x1x!tt.ptr<f16, 1>, #blocked2>
#     %19 = tt.addptr %arg2, %14 : !tt.ptr<f16, 1>, i32
#     %20 = tt.splat %19 : (!tt.ptr<f16, 1>) -> tensor<128x1x!tt.ptr<f16, 1>, #blocked3>
#     %21 = tt.addptr %arg0, %14 : !tt.ptr<f16, 1>, i32
#     %22 = tt.splat %21 : (!tt.ptr<f16, 1>) -> tensor<128x1x!tt.ptr<f16, 1>, #blocked>
#     %23 = arith.extsi %arg8 : i32 to i64
#     %24 = tt.splat %23 : (i64) -> tensor<128x1xi64, #blocked>
#     %25 = arith.extsi %arg14 : i32 to i64
#     %26 = tt.splat %25 : (i64) -> tensor<128x1xi64, #blocked3>
#     %27 = arith.extsi %arg11 : i32 to i64
#     %28 = tt.splat %27 : (i64) -> tensor<1x128xi64, #blocked2>
#     %29 = arith.extsi %arg17 : i32 to i64
#     %30 = tt.splat %29 : (i64) -> tensor<128x1xi64, #mfma>
#     %31 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
#     %32 = arith.extsi %31 : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> to tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
#     %33 = arith.addi %9, %32 : tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
#     %34 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#     %35 = arith.extsi %34 : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #mfma}>> to tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#     %36 = arith.addi %8, %35 : tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#     %37 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
#     %38 = arith.extsi %37 : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> to tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
#     %39 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked3}>>
#     %40 = arith.extsi %39 : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked3}>> to tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked3}>>
#     %41 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #mfma}>>
#     %42 = arith.extsi %41 : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #mfma}>> to tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #mfma}>>
#     %43 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
#     %44 = arith.extsi %43 : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>> to tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
#     %45 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
#     %46 = arith.extsi %45 : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>> to tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
#     %47 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked3}>>
#     %48 = arith.extsi %47 : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked3}>> to tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked3}>>
#     %49 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked1>
#     %50 = arith.addi %6, %49 : tensor<128xi32, #blocked1>
#     %51 = tt.addptr %13, %50 : tensor<128x!tt.ptr<f32, 1>, #blocked1>, tensor<128xi32, #blocked1>
#     %52 = tt.expand_dims %33 {axis = 1 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>>) -> tensor<128x1xi64, #blocked>
#     %53 = arith.muli %52, %24 : tensor<128x1xi64, #blocked>
#     %54 = tt.addptr %22, %53 : tensor<128x1x!tt.ptr<f16, 1>, #blocked>, tensor<128x1xi64, #blocked>
#     %55 = tt.expand_dims %36 {axis = 1 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #mfma}>>) -> tensor<128x1xi64, #mfma>
#     %56 = arith.muli %55, %30 : tensor<128x1xi64, #mfma>
#     %57 = tt.addptr %16, %56 : tensor<128x1x!tt.ptr<f16, 1>, #mfma>, tensor<128x1xi64, #mfma>
#     %58 = tt.broadcast %54 : (tensor<128x1x!tt.ptr<f16, 1>, #blocked>) -> tensor<128x128x!tt.ptr<f16, 1>, #blocked>
#     %59 = tt.expand_dims %38 {axis = 0 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked}>>) -> tensor<1x128xi64, #blocked>
#     %60 = tt.expand_dims %40 {axis = 0 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked3}>>) -> tensor<1x128xi64, #blocked3>
#     %61 = tt.expand_dims %42 {axis = 0 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #mfma}>>) -> tensor<1x128xi64, #mfma>
#     %62 = tt.broadcast %59 : (tensor<1x128xi64, #blocked>) -> tensor<128x128xi64, #blocked>
#     %63 = tt.addptr %58, %62 : tensor<128x128x!tt.ptr<f16, 1>, #blocked>, tensor<128x128xi64, #blocked>
#     %64 = triton_gpu.view_slice %63[0, 64] [128, 64] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked> to tensor<128x64x!tt.ptr<f16, 1>, #blocked>
#     %65 = tt.load %64 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x64xf16, #blocked>
#     %71 = triton_gpu.view_slice %63[0, 0] [128, 64] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked> to tensor<128x64x!tt.ptr<f16, 1>, #blocked>
#     %72 = tt.load %71 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x64xf16, #blocked>
#     %66 = arith.extf %65 : tensor<128x64xf16, #blocked> to tensor<128x64xf32, #blocked>
#     %67 = arith.mulf %66, %2 : tensor<128x64xf32, #blocked>
#     %68 = arith.truncf %67 : tensor<128x64xf32, #blocked> to tensor<128x64xf16, #blocked>
#     %69 = triton_gpu.convert_layout %68 : (tensor<128x64xf16, #blocked>) -> tensor<128x64xf16, #shared>
#     %70 = triton_gpu.convert_layout %69 : (tensor<128x64xf16, #shared>) -> tensor<128x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>
#     %73 = arith.extf %72 : tensor<128x64xf16, #blocked> to tensor<128x64xf32, #blocked>
#     %74 = arith.mulf %73, %3 : tensor<128x64xf32, #blocked>
#     %75 = arith.truncf %74 : tensor<128x64xf32, #blocked> to tensor<128x64xf16, #blocked>
#     %76 = triton_gpu.convert_layout %75 : (tensor<128x64xf16, #blocked>) -> tensor<128x64xf16, #shared>
#     %77 = triton_gpu.convert_layout %76 : (tensor<128x64xf16, #shared>) -> tensor<128x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>
#     %78 = tt.broadcast %60 : (tensor<1x128xi64, #blocked3>) -> tensor<128x128xi64, #blocked3>
#     %79 = tt.broadcast %61 : (tensor<1x128xi64, #mfma>) -> tensor<128x128xi64, #mfma>
#     %80 = tt.expand_dims %44 {axis = 1 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>) -> tensor<128x1xi64, #blocked2>
#     %81 = tt.addptr %18, %80 : tensor<128x1x!tt.ptr<f16, 1>, #blocked2>, tensor<128x1xi64, #blocked2>
#     %82 = tt.broadcast %81 : (tensor<128x1x!tt.ptr<f16, 1>, #blocked2>) -> tensor<128x128x!tt.ptr<f16, 1>, #blocked2>
#     %83:5 = scf.for %arg21 = %c0_i32 to %arg20 step %c128_i32 iter_args(%arg22 = %cst_1, %arg23 = %cst, %arg24 = %cst_0, %arg25 = %c0_i64, %arg26 = %c0_i64) -> (tensor<128x128xf32, #mfma>, tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>, tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>, i64, i64)  : i32 {
#       %93 = tt.splat %arg26 : (i64) -> tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
#       %94 = arith.addi %93, %46 : tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
#       %95 = tt.expand_dims %94 {axis = 0 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>) -> tensor<1x128xi64, #blocked2>
#       %96 = arith.muli %95, %28 : tensor<1x128xi64, #blocked2>
#       %97 = tt.broadcast %96 : (tensor<1x128xi64, #blocked2>) -> tensor<128x128xi64, #blocked2>
#       %98 = tt.addptr %82, %97 : tensor<128x128x!tt.ptr<f16, 1>, #blocked2>, tensor<128x128xi64, #blocked2>
#       %99 = triton_gpu.view_slice %98[0, 0] [64, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked2> to tensor<64x128x!tt.ptr<f16, 1>, #blocked2>
#       %100 = tt.load %99 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<64x128xf16, #blocked2>
#       %101 = triton_gpu.view_slice %98[64, 0] [64, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked2> to tensor<64x128x!tt.ptr<f16, 1>, #blocked2>
#       %102 = tt.load %101 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<64x128xf16, #blocked2>
#       %103 = triton_gpu.convert_layout %100 : (tensor<64x128xf16, #blocked2>) -> tensor<64x128xf16, #shared1>
#       %104 = triton_gpu.convert_layout %103 : (tensor<64x128xf16, #shared1>) -> tensor<64x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>>
#       %107 = tt.dot %77, %104, %cst_1 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>> * tensor<64x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>> -> tensor<128x128xf32, #mfma>
#       %105 = triton_gpu.convert_layout %102 : (tensor<64x128xf16, #blocked2>) -> tensor<64x128xf16, #shared1>
#       %106 = triton_gpu.convert_layout %105 : (tensor<64x128xf16, #shared1>) -> tensor<64x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>>
#       %108 = tt.dot %70, %106, %107 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>> * tensor<64x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>> -> tensor<128x128xf32, #mfma>
#       %109 = tt.splat %arg25 : (i64) -> tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked3}>>
#       %110 = arith.addi %109, %48 : tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked3}>>
#       %111 = tt.expand_dims %110 {axis = 1 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked3}>>) -> tensor<128x1xi64, #blocked3>
#       %112 = arith.muli %111, %26 : tensor<128x1xi64, #blocked3>
#       %113 = tt.addptr %20, %112 : tensor<128x1x!tt.ptr<f16, 1>, #blocked3>, tensor<128x1xi64, #blocked3>
#       %114 = tt.broadcast %113 : (tensor<128x1x!tt.ptr<f16, 1>, #blocked3>) -> tensor<128x128x!tt.ptr<f16, 1>, #blocked3>
#       %115 = tt.addptr %114, %78 : tensor<128x128x!tt.ptr<f16, 1>, #blocked3>, tensor<128x128xi64, #blocked3>
#       %116 = triton_gpu.view_slice %115[0, 0] [64, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked3> to tensor<64x128x!tt.ptr<f16, 1>, #blocked3>
#       %132 = tt.load %116 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<64x128xf16, #blocked3>
#       %133 = triton_gpu.view_slice %115[64, 0] [64, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked3> to tensor<64x128x!tt.ptr<f16, 1>, #blocked3>
#       %134 = tt.load %133 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<64x128xf16, #blocked3>
#       %117 = "tt.reduce"(%108) <{axis = 1 : i32}> ({
#       ^bb0(%arg27: f32, %arg28: f32):
#         %147 = arith.maximumf %arg27, %arg28 : f32
#         tt.reduce.return %147 : f32
#       }) : (tensor<128x128xf32, #mfma>) -> tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#       %118 = arith.maximumf %arg24, %117 : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#       %119 = arith.subf %arg24, %118 : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#       %120 = tt.extern_elementwise %119 {libname = "libdevice", libpath = "/triton/python/triton/language/../third_party/hip/lib/bitcode/cuda2gcn.bc", pure = true, symbol = "__nv_exp2f"} : (tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>) -> tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#       %121 = arith.mulf %arg23, %120 : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#       %122 = tt.expand_dims %118 {axis = 1 : i32} : (tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>) -> tensor<128x1xf32, #mfma>
#       %123 = tt.broadcast %122 : (tensor<128x1xf32, #mfma>) -> tensor<128x128xf32, #mfma>
#       %124 = arith.subf %108, %123 : tensor<128x128xf32, #mfma>
#       %125 = tt.extern_elementwise %124 {libname = "libdevice", libpath = "/triton/python/triton/language/../third_party/hip/lib/bitcode/cuda2gcn.bc", pure = true, symbol = "__nv_exp2f"} : (tensor<128x128xf32, #mfma>) -> tensor<128x128xf32, #mfma>
#       %126 = arith.truncf %125 : tensor<128x128xf32, #mfma> to tensor<128x128xf16, #mfma>
#       %127 = tt.expand_dims %120 {axis = 1 : i32} : (tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>) -> tensor<128x1xf32, #mfma>
#       %128 = tt.broadcast %127 : (tensor<128x1xf32, #mfma>) -> tensor<128x128xf32, #mfma>
#       %129 = arith.mulf %arg22, %128 : tensor<128x128xf32, #mfma>
#       %130 = triton_gpu.view_slice %126[0, 0] [128, 64] [1, 1] : tensor<128x128xf16, #mfma> to tensor<128x64xf16, #mfma>
#       %131 = triton_gpu.convert_layout %130 : (tensor<128x64xf16, #mfma>) -> tensor<128x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>
#       %135 = triton_gpu.convert_layout %132 : (tensor<64x128xf16, #blocked3>) -> tensor<64x128xf16, #shared2>
#       %136 = triton_gpu.convert_layout %135 : (tensor<64x128xf16, #shared2>) -> tensor<64x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>>
#       %139 = tt.dot %131, %136, %129 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>> * tensor<64x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>> -> tensor<128x128xf32, #mfma>
#       %140 = triton_gpu.view_slice %126[0, 64] [128, 64] [1, 1] : tensor<128x128xf16, #mfma> to tensor<128x64xf16, #mfma>
#       %141 = triton_gpu.convert_layout %140 : (tensor<128x64xf16, #mfma>) -> tensor<128x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>
#       %137 = triton_gpu.convert_layout %134 : (tensor<64x128xf16, #blocked3>) -> tensor<64x128xf16, #shared2>
#       %138 = triton_gpu.convert_layout %137 : (tensor<64x128xf16, #shared2>) -> tensor<64x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>>
#       %142 = tt.dot %141, %138, %139 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>> * tensor<64x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>> -> tensor<128x128xf32, #mfma>
#       %143 = "tt.reduce"(%125) <{axis = 1 : i32}> ({
#       ^bb0(%arg27: f32, %arg28: f32):
#         %147 = arith.addf %arg27, %arg28 : f32
#         tt.reduce.return %147 : f32
#       }) : (tensor<128x128xf32, #mfma>) -> tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#       %144 = arith.addf %121, %143 : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#       %145 = arith.addi %arg25, %c128_i64 : i64
#       %146 = arith.addi %arg26, %c128_i64 : i64
#       scf.yield %142, %144, %118, %145, %146 : tensor<128x128xf32, #mfma>, tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>, tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>, i64, i64
#     }
#     %84 = tt.extern_elementwise %83#1 {libname = "libdevice", libpath = "/triton/python/triton/language/../third_party/hip/lib/bitcode/cuda2gcn.bc", pure = true, symbol = "__nv_log2f"} : (tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>) -> tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#     %85 = arith.addf %83#2, %84 : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#     %86 = triton_gpu.convert_layout %85 : (tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>) -> tensor<128xf32, #blocked1>
#     %87 = tt.expand_dims %83#1 {axis = 1 : i32} : (tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>) -> tensor<128x1xf32, #mfma>
#     %88 = tt.broadcast %87 : (tensor<128x1xf32, #mfma>) -> tensor<128x128xf32, #mfma>
#     %89 = arith.divf %83#0, %88 : tensor<128x128xf32, #mfma>
#     %90 = arith.truncf %89 : tensor<128x128xf32, #mfma> to tensor<128x128xf16, #mfma>
#     tt.store %51, %86 {cache = 1 : i32, evict = 1 : i32} : tensor<128xf32, #blocked1>
#     %91 = tt.broadcast %57 : (tensor<128x1x!tt.ptr<f16, 1>, #mfma>) -> tensor<128x128x!tt.ptr<f16, 1>, #mfma>
#     %92 = tt.addptr %91, %79 : tensor<128x128x!tt.ptr<f16, 1>, #mfma>, tensor<128x128xi64, #mfma>
#     tt.store %92, %90 {cache = 1 : i32, evict = 1 : i32} : tensor<128x128xf16, #mfma>
#     tt.return
#   }
# }
# """
# ir = """
# #blocked = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
# #blocked1 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
# #blocked2 = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 4], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
# #blocked3 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
# #mfma = #triton_gpu.mfma<{nonKDim = 32, warpsPerCTA = [4, 1], isTransposed = true}>
# #shared = #triton_gpu.shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = false}>
# #shared1 = #triton_gpu.shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = false}>
# #shared2 = #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = false}>
# module attributes {"triton_gpu.compute-capability" = 0 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {
#   tt.func public @_attn_fwd_0d1d2d34d5d6de7de8de9c10de11de12de13c14de15de16de17c18de19de20de21c2223de24de(%arg0: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg3: f32, %arg4: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg7: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg8: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg9: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg10: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg11: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg12: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg13: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg14: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg15: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg16: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg17: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg18: i32, %arg19: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg20: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}) attributes {noinline = false} {
#     %c128_i32 = arith.constant 128 : i32
#     %c0_i32 = arith.constant 0 : i32
#     %cst = arith.constant dense<1.000000e+00> : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#     %cst_0 = arith.constant dense<0xFF800000> : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#     %cst_1 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mfma>
#     %c0_i64 = arith.constant 0 : i64
#     %c128_i64 = arith.constant 128 : i64
#     %cst_2 = arith.constant 1.44269502 : f32
#     %0 = arith.mulf %arg3, %cst_2 : f32
#     %1 = tt.splat %0 : (f32) -> tensor<128x128xf32, #blocked>
#     %2 = triton_gpu.view_slice %1[0, 64] [128, 64] [1, 1] : tensor<128x128xf32, #blocked> to tensor<128x64xf32, #blocked>
#     %3 = triton_gpu.view_slice %1[0, 0] [128, 64] [1, 1] : tensor<128x128xf32, #blocked> to tensor<128x64xf32, #blocked>
#     %4 = tt.get_program_id x : i32
#     %5 = arith.muli %4, %c128_i32 : i32
#     %6 = tt.splat %5 : (i32) -> tensor<128xi32, #blocked1>
#     %7 = arith.extsi %5 : i32 to i64
#     %8 = tt.splat %7 : (i64) -> tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#     %9 = tt.splat %7 : (i64) -> tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
#     %10 = tt.get_program_id y : i32
#     %11 = arith.muli %10, %arg20 : i32
#     %12 = tt.addptr %arg4, %11 : !tt.ptr<f32, 1>, i32
#     %13 = tt.splat %12 : (!tt.ptr<f32, 1>) -> tensor<128x!tt.ptr<f32, 1>, #blocked1>
#     %14 = arith.muli %10, %arg7 : i32
#     %15 = tt.addptr %arg5, %14 : !tt.ptr<f16, 1>, i32
#     %16 = tt.splat %15 : (!tt.ptr<f16, 1>) -> tensor<128x1x!tt.ptr<f16, 1>, #mfma>
#     %17 = tt.addptr %arg1, %14 : !tt.ptr<f16, 1>, i32
#     %18 = tt.splat %17 : (!tt.ptr<f16, 1>) -> tensor<128x1x!tt.ptr<f16, 1>, #blocked2>
#     %19 = tt.addptr %arg2, %14 : !tt.ptr<f16, 1>, i32
#     %20 = tt.splat %19 : (!tt.ptr<f16, 1>) -> tensor<128x1x!tt.ptr<f16, 1>, #blocked3>
#     %21 = tt.addptr %arg0, %14 : !tt.ptr<f16, 1>, i32
#     %22 = tt.splat %21 : (!tt.ptr<f16, 1>) -> tensor<128x1x!tt.ptr<f16, 1>, #blocked>
#     %23 = arith.extsi %arg8 : i32 to i64
#     %24 = tt.splat %23 : (i64) -> tensor<128x1xi64, #blocked>
#     %25 = arith.extsi %arg14 : i32 to i64
#     %26 = tt.splat %25 : (i64) -> tensor<128x1xi64, #blocked3>
#     %27 = arith.extsi %arg11 : i32 to i64
#     %28 = tt.splat %27 : (i64) -> tensor<1x128xi64, #blocked2>
#     %29 = arith.extsi %arg17 : i32 to i64
#     %30 = tt.splat %29 : (i64) -> tensor<128x1xi64, #mfma>
#     %31 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
#     %32 = arith.extsi %31 : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> to tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
#     %33 = arith.addi %9, %32 : tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
#     %34 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#     %35 = arith.extsi %34 : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #mfma}>> to tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#     %36 = arith.addi %8, %35 : tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#     %37 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
#     %38 = arith.extsi %37 : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> to tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
#     %39 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked3}>>
#     %40 = arith.extsi %39 : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked3}>> to tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked3}>>
#     %41 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #mfma}>>
#     %42 = arith.extsi %41 : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #mfma}>> to tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #mfma}>>
#     %43 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
#     %44 = arith.extsi %43 : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>> to tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
#     %45 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
#     %46 = arith.extsi %45 : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>> to tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
#     %47 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked3}>>
#     %48 = arith.extsi %47 : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked3}>> to tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked3}>>
#     %49 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked1>
#     %50 = arith.addi %6, %49 : tensor<128xi32, #blocked1>
#     %51 = tt.addptr %13, %50 : tensor<128x!tt.ptr<f32, 1>, #blocked1>, tensor<128xi32, #blocked1>
#     %52 = tt.expand_dims %33 {axis = 1 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>>) -> tensor<128x1xi64, #blocked>
#     %53 = arith.muli %52, %24 : tensor<128x1xi64, #blocked>
#     %54 = tt.addptr %22, %53 : tensor<128x1x!tt.ptr<f16, 1>, #blocked>, tensor<128x1xi64, #blocked>
#     %55 = tt.expand_dims %36 {axis = 1 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #mfma}>>) -> tensor<128x1xi64, #mfma>
#     %56 = arith.muli %55, %30 : tensor<128x1xi64, #mfma>
#     %57 = tt.addptr %16, %56 : tensor<128x1x!tt.ptr<f16, 1>, #mfma>, tensor<128x1xi64, #mfma>
#     %58 = tt.broadcast %54 : (tensor<128x1x!tt.ptr<f16, 1>, #blocked>) -> tensor<128x128x!tt.ptr<f16, 1>, #blocked>
#     %59 = tt.expand_dims %38 {axis = 0 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked}>>) -> tensor<1x128xi64, #blocked>
#     %60 = tt.expand_dims %40 {axis = 0 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked3}>>) -> tensor<1x128xi64, #blocked3>
#     %61 = tt.expand_dims %42 {axis = 0 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #mfma}>>) -> tensor<1x128xi64, #mfma>
#     %62 = tt.broadcast %59 : (tensor<1x128xi64, #blocked>) -> tensor<128x128xi64, #blocked>
#     %63 = tt.addptr %58, %62 : tensor<128x128x!tt.ptr<f16, 1>, #blocked>, tensor<128x128xi64, #blocked>
#     %64 = triton_gpu.view_slice %63[0, 64] [128, 64] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked> to tensor<128x64x!tt.ptr<f16, 1>, #blocked>
#     %65 = tt.load %64 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x64xf16, #blocked>
#     %71 = triton_gpu.view_slice %63[0, 0] [128, 64] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked> to tensor<128x64x!tt.ptr<f16, 1>, #blocked>
#     %72 = tt.load %71 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x64xf16, #blocked>
#     %66 = arith.extf %65 : tensor<128x64xf16, #blocked> to tensor<128x64xf32, #blocked>
#     %67 = arith.mulf %66, %2 : tensor<128x64xf32, #blocked>
#     %68 = arith.truncf %67 : tensor<128x64xf32, #blocked> to tensor<128x64xf16, #blocked>
#     %69 = triton_gpu.convert_layout %68 : (tensor<128x64xf16, #blocked>) -> tensor<128x64xf16, #shared>
#     %70 = triton_gpu.convert_layout %69 : (tensor<128x64xf16, #shared>) -> tensor<128x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>
#     %73 = arith.extf %72 : tensor<128x64xf16, #blocked> to tensor<128x64xf32, #blocked>
#     %74 = arith.mulf %73, %3 : tensor<128x64xf32, #blocked>
#     %75 = arith.truncf %74 : tensor<128x64xf32, #blocked> to tensor<128x64xf16, #blocked>
#     %76 = triton_gpu.convert_layout %75 : (tensor<128x64xf16, #blocked>) -> tensor<128x64xf16, #shared>
#     %77 = triton_gpu.convert_layout %76 : (tensor<128x64xf16, #shared>) -> tensor<128x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>
#     %78 = tt.broadcast %60 : (tensor<1x128xi64, #blocked3>) -> tensor<128x128xi64, #blocked3>
#     %79 = tt.broadcast %61 : (tensor<1x128xi64, #mfma>) -> tensor<128x128xi64, #mfma>
#     %80 = tt.expand_dims %44 {axis = 1 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>) -> tensor<128x1xi64, #blocked2>
#     %81 = tt.addptr %18, %80 : tensor<128x1x!tt.ptr<f16, 1>, #blocked2>, tensor<128x1xi64, #blocked2>
#     %82 = tt.broadcast %81 : (tensor<128x1x!tt.ptr<f16, 1>, #blocked2>) -> tensor<128x128x!tt.ptr<f16, 1>, #blocked2>
#     %170 = tt.splat %c0_i64 : (i64) -> tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
#     %171 = arith.addi %170, %46 : tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
#     %172 = tt.expand_dims %171 {axis = 0 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>) -> tensor<1x128xi64, #blocked2>
#     %173 = arith.muli %172, %28 : tensor<1x128xi64, #blocked2>
#     %174 = tt.broadcast %173 : (tensor<1x128xi64, #blocked2>) -> tensor<128x128xi64, #blocked2>
#     %175 = tt.addptr %82, %174 : tensor<128x128x!tt.ptr<f16, 1>, #blocked2>, tensor<128x128xi64, #blocked2>
#     %176 = triton_gpu.view_slice %175[0, 0] [64, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked2> to tensor<64x128x!tt.ptr<f16, 1>, #blocked2>
#     %177 = tt.load %176 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<64x128xf16, #blocked2>
#     %178 = triton_gpu.convert_layout %177 : (tensor<64x128xf16, #blocked2>) -> tensor<64x128xf16, #shared1>
#     %179 = arith.addi %c0_i64, %c128_i64 : i64
#     %200 = arith.subi %arg20, %c128_i32 : i32
#     %83:6 = scf.for %arg21 = %c0_i32 to %200 step %c128_i32 iter_args(%arg22 = %cst_1, %arg23 = %cst, %arg24 = %cst_0, %arg25 = %c0_i64, %arg26 = %178, %arg27 = %179) -> (tensor<128x128xf32, #mfma>, tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>, tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>, i64, tensor<64x128xf16, #shared1>, i64)  : i32 {
#       %180 = tt.splat %arg27 : (i64) -> tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
#       %181 = arith.addi %180, %46 : tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
#       %182 = tt.expand_dims %181 {axis = 0 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>) -> tensor<1x128xi64, #blocked2>
#       %183 = arith.muli %182, %28 : tensor<1x128xi64, #blocked2>
#       %184 = tt.broadcast %183 : (tensor<1x128xi64, #blocked2>) -> tensor<128x128xi64, #blocked2>
#       %185 = tt.addptr %82, %184 : tensor<128x128x!tt.ptr<f16, 1>, #blocked2>, tensor<128x128xi64, #blocked2>
#       %186 = triton_gpu.view_slice %185[0, 0] [64, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked2> to tensor<64x128x!tt.ptr<f16, 1>, #blocked2>
#       %187 = tt.load %186 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<64x128xf16, #blocked2>
#       %188 = triton_gpu.convert_layout %187 : (tensor<64x128xf16, #blocked2>) -> tensor<64x128xf16, #shared1>
#       %189 = tt.splat %arg25 : (i64) -> tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
#       %190 = arith.addi %189, %46 : tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
#       %191 = tt.expand_dims %190 {axis = 0 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>) -> tensor<1x128xi64, #blocked2>
#       %192 = arith.muli %191, %28 : tensor<1x128xi64, #blocked2>
#       %193 = tt.broadcast %192 : (tensor<1x128xi64, #blocked2>) -> tensor<128x128xi64, #blocked2>
#       %194 = tt.addptr %82, %193 : tensor<128x128x!tt.ptr<f16, 1>, #blocked2>, tensor<128x128xi64, #blocked2>
#       %101 = triton_gpu.view_slice %194[64, 0] [64, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked2> to tensor<64x128x!tt.ptr<f16, 1>, #blocked2>
#       %104 = triton_gpu.convert_layout %arg26 : (tensor<64x128xf16, #shared1>) -> tensor<64x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>>                                          
#       %107 = tt.dot %77, %104, %cst_1 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>> * tensor<64x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>> -> tensor<128x128xf32, #mfma>
#       %102 = tt.load %101 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<64x128xf16, #blocked2>
#       %105 = triton_gpu.convert_layout %102 : (tensor<64x128xf16, #blocked2>) -> tensor<64x128xf16, #shared1>
#       %106 = triton_gpu.convert_layout %105 : (tensor<64x128xf16, #shared1>) -> tensor<64x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>>
#       %108 = tt.dot %70, %106, %107 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>> * tensor<64x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>> -> tensor<128x128xf32, #mfma>
#       %117 = "tt.reduce"(%108) <{axis = 1 : i32}> ({
#       ^bb0(%arg29: f32, %arg30: f32):
#         %147 = arith.maximumf %arg29, %arg30 : f32
#         tt.reduce.return %147 : f32
#       }) : (tensor<128x128xf32, #mfma>) -> tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#       %118 = arith.maximumf %arg24, %117 : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#       %119 = arith.subf %arg24, %118 : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#       %120 = tt.extern_elementwise %119 {libname = "libdevice", libpath = "/triton/python/triton/language/../third_party/hip/lib/bitcode/cuda2gcn.bc", pure = true, symbol = "__nv_exp2f"} : (tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>) -> tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#       %121 = arith.mulf %arg23, %120 : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#       %122 = tt.expand_dims %118 {axis = 1 : i32} : (tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>) -> tensor<128x1xf32, #mfma>
#       %123 = tt.broadcast %122 : (tensor<128x1xf32, #mfma>) -> tensor<128x128xf32, #mfma>
#       %124 = arith.subf %108, %123 : tensor<128x128xf32, #mfma>
#       %125 = tt.extern_elementwise %124 {libname = "libdevice", libpath = "/triton/python/triton/language/../third_party/hip/lib/bitcode/cuda2gcn.bc", pure = true, symbol = "__nv_exp2f"} : (tensor<128x128xf32, #mfma>) -> tensor<128x128xf32, #mfma>
#       %126 = arith.truncf %125 : tensor<128x128xf32, #mfma> to tensor<128x128xf16, #mfma>
#       %127 = tt.expand_dims %120 {axis = 1 : i32} : (tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>) -> tensor<128x1xf32, #mfma>
#       %128 = tt.broadcast %127 : (tensor<128x1xf32, #mfma>) -> tensor<128x128xf32, #mfma>
#       %129 = arith.mulf %arg22, %128 : tensor<128x128xf32, #mfma>
#       %109 = tt.splat %arg25 : (i64) -> tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked3}>>
#       %110 = arith.addi %109, %48 : tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked3}>>
#       %111 = tt.expand_dims %110 {axis = 1 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked3}>>) -> tensor<128x1xi64, #blocked3>
#       %112 = arith.muli %111, %26 : tensor<128x1xi64, #blocked3>
#       %113 = tt.addptr %20, %112 : tensor<128x1x!tt.ptr<f16, 1>, #blocked3>, tensor<128x1xi64, #blocked3>
#       %114 = tt.broadcast %113 : (tensor<128x1x!tt.ptr<f16, 1>, #blocked3>) -> tensor<128x128x!tt.ptr<f16, 1>, #blocked3>
#       %115 = tt.addptr %114, %78 : tensor<128x128x!tt.ptr<f16, 1>, #blocked3>, tensor<128x128xi64, #blocked3>
#       %130 = triton_gpu.view_slice %126[0, 0] [128, 64] [1, 1] : tensor<128x128xf16, #mfma> to tensor<128x64xf16, #mfma>
#       %116 = triton_gpu.view_slice %115[0, 0] [64, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked3> to tensor<64x128x!tt.ptr<f16, 1>, #blocked3>
#       %132 = tt.load %116 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<64x128xf16, #blocked3>
#       %133 = triton_gpu.view_slice %115[64, 0] [64, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked3> to tensor<64x128x!tt.ptr<f16, 1>, #blocked3>
#       %134 = tt.load %133 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<64x128xf16, #blocked3>
#       %131 = triton_gpu.convert_layout %130 : (tensor<128x64xf16, #mfma>) -> tensor<128x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>
#       %135 = triton_gpu.convert_layout %132 : (tensor<64x128xf16, #blocked3>) -> tensor<64x128xf16, #shared2>
#       %136 = triton_gpu.convert_layout %135 : (tensor<64x128xf16, #shared2>) -> tensor<64x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>>
#       %139 = tt.dot %131, %136, %129 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>> * tensor<64x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>> -> tensor<128x128xf32, #mfma>
#       %140 = triton_gpu.view_slice %126[0, 64] [128, 64] [1, 1] : tensor<128x128xf16, #mfma> to tensor<128x64xf16, #mfma>
#       %141 = triton_gpu.convert_layout %140 : (tensor<128x64xf16, #mfma>) -> tensor<128x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>
#       %137 = triton_gpu.convert_layout %134 : (tensor<64x128xf16, #blocked3>) -> tensor<64x128xf16, #shared2>
#       %138 = triton_gpu.convert_layout %137 : (tensor<64x128xf16, #shared2>) -> tensor<64x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>>
#       %142 = tt.dot %141, %138, %139 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>> * tensor<64x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>> -> tensor<128x128xf32, #mfma>
#       %143 = "tt.reduce"(%125) <{axis = 1 : i32}> ({
#       ^bb0(%arg29: f32, %arg30: f32):
#         %147 = arith.addf %arg29, %arg30 : f32
#         tt.reduce.return %147 : f32
#       }) : (tensor<128x128xf32, #mfma>) -> tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#       %144 = arith.addf %121, %143 : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#       %145 = arith.addi %arg25, %c128_i64 : i64
#       %148 = arith.addi %arg27, %c128_i64 : i64
#       scf.yield %142, %144, %118, %145, %188, %148 : tensor<128x128xf32, #mfma>, tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>, tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>, i64, tensor<64x128xf16, #shared1>, i64
#     }
#     %84 = tt.extern_elementwise %83#1 {libname = "libdevice", libpath = "/triton/python/triton/language/../third_party/hip/lib/bitcode/cuda2gcn.bc", pure = true, symbol = "__nv_log2f"} : (tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>) -> tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#     %85 = arith.addf %83#2, %84 : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#     %86 = triton_gpu.convert_layout %85 : (tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>) -> tensor<128xf32, #blocked1>
#     %87 = tt.expand_dims %83#1 {axis = 1 : i32} : (tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>) -> tensor<128x1xf32, #mfma>
#     %88 = tt.broadcast %87 : (tensor<128x1xf32, #mfma>) -> tensor<128x128xf32, #mfma>
#     %89 = arith.divf %83#0, %88 : tensor<128x128xf32, #mfma>
#     %90 = arith.truncf %89 : tensor<128x128xf32, #mfma> to tensor<128x128xf16, #mfma>
#     tt.store %51, %86 {cache = 1 : i32, evict = 1 : i32} : tensor<128xf32, #blocked1>
#     %91 = tt.broadcast %57 : (tensor<128x1x!tt.ptr<f16, 1>, #mfma>) -> tensor<128x128x!tt.ptr<f16, 1>, #mfma>
#     %92 = tt.addptr %91, %79 : tensor<128x128x!tt.ptr<f16, 1>, #mfma>, tensor<128x128xi64, #mfma>
#     tt.store %92, %90 {cache = 1 : i32, evict = 1 : i32} : tensor<128x128xf16, #mfma>
#     tt.return
#   }
# }
# """

ir = """
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 16], threadsPerWarp = [32, 2], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
#blocked2 = #triton_gpu.blocked<{sizePerThread = [16, 1], threadsPerWarp = [2, 32], warpsPerCTA = [1, 4], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked3 = #triton_gpu.blocked<{sizePerThread = [1, 16], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#mfma = #triton_gpu.mfma<{nonKDim = 32, warpsPerCTA = [4, 1], isTransposed = true}>
#shared = #triton_gpu.shared<{vec = 4, perPhase = 2, maxPhase = 8, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = false}>
#shared1 = #triton_gpu.shared<{vec = 4, perPhase = 2, maxPhase = 8, order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = false}>
#shared2 = #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = false}>
module attributes {"triton_gpu.compute-capability" = 0 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {
  tt.func public @_attn_fwd_0d1d2d34d5d6de7de8de9c10de11de12de13c14de15de16de17c18de19de20de21c2223de24de(%arg0: !tt.ptr<f8E5M2, 1> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f8E5M2, 1> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg3: f32, %arg4: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg7: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg8: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg9: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg10: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg11: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg12: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg13: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg14: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg15: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg16: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg17: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg18: i32, %arg19: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg20: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}) attributes {noinline = false} {
    %c128_i32 = arith.constant 128 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<1.000000e+00> : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
    %cst_0 = arith.constant dense<0xFF800000> : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mfma>
    %c0_i64 = arith.constant 0 : i64
    %c128_i64 = arith.constant 128 : i64
    %cst_2 = arith.constant 1.44269502 : f32
    %0 = arith.mulf %arg3, %cst_2 : f32
    %1 = tt.splat %0 : (f32) -> tensor<128x128xf32, #blocked>
    %2 = triton_gpu.view_slice %1[0, 96] [128, 32] [1, 1] : tensor<128x128xf32, #blocked> to tensor<128x32xf32, #blocked>
    %3 = triton_gpu.view_slice %1[0, 64] [128, 32] [1, 1] : tensor<128x128xf32, #blocked> to tensor<128x32xf32, #blocked>
    %4 = triton_gpu.view_slice %1[0, 32] [128, 32] [1, 1] : tensor<128x128xf32, #blocked> to tensor<128x32xf32, #blocked>
    %5 = triton_gpu.view_slice %1[0, 0] [128, 32] [1, 1] : tensor<128x128xf32, #blocked> to tensor<128x32xf32, #blocked>
    %6 = tt.get_program_id x : i32
    %7 = arith.muli %6, %c128_i32 : i32
    %8 = tt.splat %7 : (i32) -> tensor<128xi32, #blocked1>
    %9 = arith.extsi %7 : i32 to i64
    %10 = tt.splat %9 : (i64) -> tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
    %11 = tt.splat %9 : (i64) -> tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %12 = tt.get_program_id y : i32
    %13 = arith.muli %12, %arg20 : i32
    %14 = tt.addptr %arg4, %13 : !tt.ptr<f32, 1>, i32
    %15 = tt.splat %14 : (!tt.ptr<f32, 1>) -> tensor<128x!tt.ptr<f32, 1>, #blocked1>
    %16 = arith.muli %12, %arg7 : i32
    %17 = tt.addptr %arg5, %16 : !tt.ptr<f16, 1>, i32
    %18 = tt.splat %17 : (!tt.ptr<f16, 1>) -> tensor<128x1x!tt.ptr<f16, 1>, #mfma>
    %19 = tt.addptr %arg1, %16 : !tt.ptr<f8E5M2, 1>, i32
    %20 = tt.splat %19 : (!tt.ptr<f8E5M2, 1>) -> tensor<128x1x!tt.ptr<f8E5M2, 1>, #blocked2>
    %21 = tt.addptr %arg2, %16 : !tt.ptr<f16, 1>, i32
    %22 = tt.splat %21 : (!tt.ptr<f16, 1>) -> tensor<128x1x!tt.ptr<f16, 1>, #blocked3>
    %23 = tt.addptr %arg0, %16 : !tt.ptr<f8E5M2, 1>, i32
    %24 = tt.splat %23 : (!tt.ptr<f8E5M2, 1>) -> tensor<128x1x!tt.ptr<f8E5M2, 1>, #blocked>
    %25 = arith.extsi %arg8 : i32 to i64
    %26 = tt.splat %25 : (i64) -> tensor<128x1xi64, #blocked>
    %27 = arith.extsi %arg14 : i32 to i64
    %28 = tt.splat %27 : (i64) -> tensor<128x1xi64, #blocked3>
    %29 = arith.extsi %arg11 : i32 to i64
    %30 = tt.splat %29 : (i64) -> tensor<1x128xi64, #blocked2>
    %31 = arith.extsi %arg17 : i32 to i64
    %32 = tt.splat %31 : (i64) -> tensor<128x1xi64, #mfma>
    %33 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %34 = arith.extsi %33 : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> to tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %35 = arith.addi %11, %34 : tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %36 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
    %37 = arith.extsi %36 : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #mfma}>> to tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
    %38 = arith.addi %10, %37 : tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
    %39 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %40 = arith.extsi %39 : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> to tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %41 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked3}>>
    %42 = arith.extsi %41 : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked3}>> to tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked3}>>
    %43 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #mfma}>>
    %44 = arith.extsi %43 : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #mfma}>> to tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #mfma}>>
    %45 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %46 = arith.extsi %45 : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>> to tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %47 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
    %48 = arith.extsi %47 : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>> to tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
    %49 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked3}>>
    %50 = arith.extsi %49 : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked3}>> to tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked3}>>
    %51 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked1>
    %52 = arith.addi %8, %51 : tensor<128xi32, #blocked1>
    %53 = tt.addptr %15, %52 : tensor<128x!tt.ptr<f32, 1>, #blocked1>, tensor<128xi32, #blocked1>
    %54 = tt.expand_dims %35 {axis = 1 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>>) -> tensor<128x1xi64, #blocked>
    %55 = arith.muli %54, %26 : tensor<128x1xi64, #blocked>
    %56 = tt.addptr %24, %55 : tensor<128x1x!tt.ptr<f8E5M2, 1>, #blocked>, tensor<128x1xi64, #blocked>
    %57 = tt.expand_dims %38 {axis = 1 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #mfma}>>) -> tensor<128x1xi64, #mfma>
    %58 = arith.muli %57, %32 : tensor<128x1xi64, #mfma>
    %59 = tt.addptr %18, %58 : tensor<128x1x!tt.ptr<f16, 1>, #mfma>, tensor<128x1xi64, #mfma>
    %60 = tt.broadcast %56 : (tensor<128x1x!tt.ptr<f8E5M2, 1>, #blocked>) -> tensor<128x128x!tt.ptr<f8E5M2, 1>, #blocked>
    %61 = tt.expand_dims %40 {axis = 0 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked}>>) -> tensor<1x128xi64, #blocked>
    %62 = tt.expand_dims %42 {axis = 0 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked3}>>) -> tensor<1x128xi64, #blocked3>
    %63 = tt.expand_dims %44 {axis = 0 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #mfma}>>) -> tensor<1x128xi64, #mfma>
    %64 = tt.broadcast %61 : (tensor<1x128xi64, #blocked>) -> tensor<128x128xi64, #blocked>
    %65 = tt.addptr %60, %64 : tensor<128x128x!tt.ptr<f8E5M2, 1>, #blocked>, tensor<128x128xi64, #blocked>
    %66 = triton_gpu.view_slice %65[0, 96] [128, 32] [1, 1] : tensor<128x128x!tt.ptr<f8E5M2, 1>, #blocked> to tensor<128x32x!tt.ptr<f8E5M2, 1>, #blocked>
    %67 = tt.load %66 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x32xf8E5M2, #blocked>
    %68 = tt.fp_to_fp %67 : tensor<128x32xf8E5M2, #blocked> -> tensor<128x32xf32, #blocked>
    %69 = arith.mulf %68, %2 : tensor<128x32xf32, #blocked>
    %70 = tt.fp_to_fp %69 : tensor<128x32xf32, #blocked> -> tensor<128x32xf8E5M2, #blocked>
    %71 = tt.fp_to_fp %70 : tensor<128x32xf8E5M2, #blocked> -> tensor<128x32xf16, #blocked>
    %72 = triton_gpu.convert_layout %71 : (tensor<128x32xf16, #blocked>) -> tensor<128x32xf16, #shared>
    %73 = triton_gpu.convert_layout %72 : (tensor<128x32xf16, #shared>) -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>
    %74 = triton_gpu.view_slice %65[0, 64] [128, 32] [1, 1] : tensor<128x128x!tt.ptr<f8E5M2, 1>, #blocked> to tensor<128x32x!tt.ptr<f8E5M2, 1>, #blocked>
    %75 = tt.load %74 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x32xf8E5M2, #blocked>
    %76 = tt.fp_to_fp %75 : tensor<128x32xf8E5M2, #blocked> -> tensor<128x32xf32, #blocked>
    %77 = arith.mulf %76, %3 : tensor<128x32xf32, #blocked>
    %78 = tt.fp_to_fp %77 : tensor<128x32xf32, #blocked> -> tensor<128x32xf8E5M2, #blocked>
    %79 = tt.fp_to_fp %78 : tensor<128x32xf8E5M2, #blocked> -> tensor<128x32xf16, #blocked>
    %80 = triton_gpu.convert_layout %79 : (tensor<128x32xf16, #blocked>) -> tensor<128x32xf16, #shared>
    %81 = triton_gpu.convert_layout %80 : (tensor<128x32xf16, #shared>) -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>
    %82 = triton_gpu.view_slice %65[0, 32] [128, 32] [1, 1] : tensor<128x128x!tt.ptr<f8E5M2, 1>, #blocked> to tensor<128x32x!tt.ptr<f8E5M2, 1>, #blocked>
    %83 = tt.load %82 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x32xf8E5M2, #blocked>
    %84 = tt.fp_to_fp %83 : tensor<128x32xf8E5M2, #blocked> -> tensor<128x32xf32, #blocked>
    %85 = arith.mulf %84, %4 : tensor<128x32xf32, #blocked>
    %86 = tt.fp_to_fp %85 : tensor<128x32xf32, #blocked> -> tensor<128x32xf8E5M2, #blocked>
    %87 = tt.fp_to_fp %86 : tensor<128x32xf8E5M2, #blocked> -> tensor<128x32xf16, #blocked>
    %88 = triton_gpu.convert_layout %87 : (tensor<128x32xf16, #blocked>) -> tensor<128x32xf16, #shared>
    %89 = triton_gpu.convert_layout %88 : (tensor<128x32xf16, #shared>) -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>
    %90 = triton_gpu.view_slice %65[0, 0] [128, 32] [1, 1] : tensor<128x128x!tt.ptr<f8E5M2, 1>, #blocked> to tensor<128x32x!tt.ptr<f8E5M2, 1>, #blocked>
    %91 = tt.load %90 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x32xf8E5M2, #blocked>
    %92 = tt.fp_to_fp %91 : tensor<128x32xf8E5M2, #blocked> -> tensor<128x32xf32, #blocked>
    %93 = arith.mulf %92, %5 : tensor<128x32xf32, #blocked>
    %94 = tt.fp_to_fp %93 : tensor<128x32xf32, #blocked> -> tensor<128x32xf8E5M2, #blocked>
    %95 = tt.fp_to_fp %94 : tensor<128x32xf8E5M2, #blocked> -> tensor<128x32xf16, #blocked>
    %96 = triton_gpu.convert_layout %95 : (tensor<128x32xf16, #blocked>) -> tensor<128x32xf16, #shared>
    %97 = triton_gpu.convert_layout %96 : (tensor<128x32xf16, #shared>) -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>
    %98 = tt.broadcast %62 : (tensor<1x128xi64, #blocked3>) -> tensor<128x128xi64, #blocked3>
    %99 = tt.broadcast %63 : (tensor<1x128xi64, #mfma>) -> tensor<128x128xi64, #mfma>
    %100 = tt.expand_dims %46 {axis = 1 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>) -> tensor<128x1xi64, #blocked2>
    %101 = tt.addptr %20, %100 : tensor<128x1x!tt.ptr<f8E5M2, 1>, #blocked2>, tensor<128x1xi64, #blocked2>
    %102 = tt.broadcast %101 : (tensor<128x1x!tt.ptr<f8E5M2, 1>, #blocked2>) -> tensor<128x128x!tt.ptr<f8E5M2, 1>, #blocked2>
    %103:5 = scf.for %arg21 = %c0_i32 to %arg20 step %c128_i32 iter_args(%arg22 = %cst_1, %arg23 = %cst, %arg24 = %cst_0, %arg25 = %c0_i64, %arg26 = %c0_i64) -> (tensor<128x128xf32, #mfma>, tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>, tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>, i64, i64)  : i32 {
      %113 = tt.splat %arg26 : (i64) -> tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
      %114 = arith.addi %113, %48 : tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
      %115 = tt.expand_dims %114 {axis = 0 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>) -> tensor<1x128xi64, #blocked2>
      %116 = arith.muli %115, %30 : tensor<1x128xi64, #blocked2>
      %117 = tt.broadcast %116 : (tensor<1x128xi64, #blocked2>) -> tensor<128x128xi64, #blocked2>
      %118 = tt.addptr %102, %117 : tensor<128x128x!tt.ptr<f8E5M2, 1>, #blocked2>, tensor<128x128xi64, #blocked2>
      %119 = triton_gpu.view_slice %118[64, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f8E5M2, 1>, #blocked2> to tensor<32x128x!tt.ptr<f8E5M2, 1>, #blocked2>
      %120 = triton_gpu.view_slice %118[0, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f8E5M2, 1>, #blocked2> to tensor<32x128x!tt.ptr<f8E5M2, 1>, #blocked2>
      %121 = tt.load %120 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf8E5M2, #blocked2>
      %122 = triton_gpu.view_slice %118[32, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f8E5M2, 1>, #blocked2> to tensor<32x128x!tt.ptr<f8E5M2, 1>, #blocked2>
      %123 = tt.load %122 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf8E5M2, #blocked2>
      %124 = tt.fp_to_fp %121 : tensor<32x128xf8E5M2, #blocked2> -> tensor<32x128xf16, #blocked2>
      %125 = triton_gpu.convert_layout %124 : (tensor<32x128xf16, #blocked2>) -> tensor<32x128xf16, #shared1>
      %126 = triton_gpu.convert_layout %125 : (tensor<32x128xf16, #shared1>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>>
      %127 = tt.fp_to_fp %123 : tensor<32x128xf8E5M2, #blocked2> -> tensor<32x128xf16, #blocked2>
      %128 = triton_gpu.convert_layout %127 : (tensor<32x128xf16, #blocked2>) -> tensor<32x128xf16, #shared1>
      %129 = triton_gpu.convert_layout %128 : (tensor<32x128xf16, #shared1>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>>
      %130 = tt.dot %97, %126, %cst_1 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>> -> tensor<128x128xf32, #mfma>
      %131 = tt.load %119 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf8E5M2, #blocked2>
      %132 = triton_gpu.view_slice %118[96, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f8E5M2, 1>, #blocked2> to tensor<32x128x!tt.ptr<f8E5M2, 1>, #blocked2>
      %133 = tt.load %132 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf8E5M2, #blocked2>
      %134 = tt.fp_to_fp %131 : tensor<32x128xf8E5M2, #blocked2> -> tensor<32x128xf16, #blocked2>
      %135 = triton_gpu.convert_layout %134 : (tensor<32x128xf16, #blocked2>) -> tensor<32x128xf16, #shared1>
      %136 = triton_gpu.convert_layout %135 : (tensor<32x128xf16, #shared1>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>>
      %137 = tt.dot %89, %129, %130 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>> -> tensor<128x128xf32, #mfma>
      %138 = tt.fp_to_fp %133 : tensor<32x128xf8E5M2, #blocked2> -> tensor<32x128xf16, #blocked2>
      %139 = triton_gpu.convert_layout %138 : (tensor<32x128xf16, #blocked2>) -> tensor<32x128xf16, #shared1>
      %140 = triton_gpu.convert_layout %139 : (tensor<32x128xf16, #shared1>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>>
      %141 = tt.dot %81, %136, %137 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>> -> tensor<128x128xf32, #mfma>
      %142 = tt.dot %73, %140, %141 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>> -> tensor<128x128xf32, #mfma>
      %143 = "tt.reduce"(%142) <{axis = 1 : i32}> ({
      ^bb0(%arg27: f32, %arg28: f32):
        %195 = arith.maximumf %arg27, %arg28 : f32
        tt.reduce.return %195 : f32
      }) : (tensor<128x128xf32, #mfma>) -> tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
      %144 = arith.maximumf %arg24, %143 : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
      %145 = arith.subf %arg24, %144 : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
      %146 = tt.extern_elementwise %145 {libname = "libdevice", libpath = "/triton/python/triton/language/../third_party/hip/lib/bitcode/cuda2gcn.bc", pure = true, symbol = "__nv_exp2f"} : (tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>) -> tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
      %147 = arith.mulf %arg23, %146 : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
      %148 = tt.expand_dims %144 {axis = 1 : i32} : (tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>) -> tensor<128x1xf32, #mfma>
      %149 = tt.broadcast %148 : (tensor<128x1xf32, #mfma>) -> tensor<128x128xf32, #mfma>
      %150 = arith.subf %142, %149 : tensor<128x128xf32, #mfma>
      %151 = tt.extern_elementwise %150 {libname = "libdevice", libpath = "/triton/python/triton/language/../third_party/hip/lib/bitcode/cuda2gcn.bc", pure = true, symbol = "__nv_exp2f"} : (tensor<128x128xf32, #mfma>) -> tensor<128x128xf32, #mfma>
      %152 = arith.truncf %151 : tensor<128x128xf32, #mfma> to tensor<128x128xf16, #mfma>
      %153 = tt.expand_dims %146 {axis = 1 : i32} : (tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>) -> tensor<128x1xf32, #mfma>
      %154 = tt.broadcast %153 : (tensor<128x1xf32, #mfma>) -> tensor<128x128xf32, #mfma>
      %155 = arith.mulf %arg22, %154 : tensor<128x128xf32, #mfma>
      %156 = tt.splat %arg25 : (i64) -> tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked3}>>
      %157 = arith.addi %156, %50 : tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked3}>>
      %158 = tt.expand_dims %157 {axis = 1 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked3}>>) -> tensor<128x1xi64, #blocked3>
      %159 = arith.muli %158, %28 : tensor<128x1xi64, #blocked3>
      %160 = tt.addptr %22, %159 : tensor<128x1x!tt.ptr<f16, 1>, #blocked3>, tensor<128x1xi64, #blocked3>
      %161 = tt.broadcast %160 : (tensor<128x1x!tt.ptr<f16, 1>, #blocked3>) -> tensor<128x128x!tt.ptr<f16, 1>, #blocked3>
      %162 = tt.addptr %161, %98 : tensor<128x128x!tt.ptr<f16, 1>, #blocked3>, tensor<128x128xi64, #blocked3>
      %163 = triton_gpu.view_slice %162[64, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked3> to tensor<32x128x!tt.ptr<f16, 1>, #blocked3>
      %164 = triton_gpu.view_slice %162[0, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked3> to tensor<32x128x!tt.ptr<f16, 1>, #blocked3>
      %165 = triton_gpu.view_slice %152[0, 0] [128, 32] [1, 1] : tensor<128x128xf16, #mfma> to tensor<128x32xf16, #mfma>
      %166 = triton_gpu.convert_layout %165 : (tensor<128x32xf16, #mfma>) -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>
      %167 = tt.load %164 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #blocked3>
      %168 = triton_gpu.view_slice %162[32, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked3> to tensor<32x128x!tt.ptr<f16, 1>, #blocked3>
      %169 = tt.load %168 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #blocked3>
      %170 = triton_gpu.convert_layout %167 : (tensor<32x128xf16, #blocked3>) -> tensor<32x128xf16, #shared2>
      %171 = triton_gpu.convert_layout %170 : (tensor<32x128xf16, #shared2>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>>
      %172 = triton_gpu.convert_layout %169 : (tensor<32x128xf16, #blocked3>) -> tensor<32x128xf16, #shared2>
      %173 = triton_gpu.convert_layout %172 : (tensor<32x128xf16, #shared2>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>>
      %174 = tt.dot %166, %171, %155 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>> -> tensor<128x128xf32, #mfma>
      %175 = triton_gpu.view_slice %152[0, 32] [128, 32] [1, 1] : tensor<128x128xf16, #mfma> to tensor<128x32xf16, #mfma>
      %176 = triton_gpu.convert_layout %175 : (tensor<128x32xf16, #mfma>) -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>
      %177 = tt.load %163 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #blocked3>
      %178 = triton_gpu.view_slice %162[96, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked3> to tensor<32x128x!tt.ptr<f16, 1>, #blocked3>
      %179 = tt.load %178 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #blocked3>
      %180 = triton_gpu.convert_layout %177 : (tensor<32x128xf16, #blocked3>) -> tensor<32x128xf16, #shared2>
      %181 = triton_gpu.convert_layout %180 : (tensor<32x128xf16, #shared2>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>>
      %182 = tt.dot %176, %173, %174 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>> -> tensor<128x128xf32, #mfma>
      %183 = triton_gpu.view_slice %152[0, 64] [128, 32] [1, 1] : tensor<128x128xf16, #mfma> to tensor<128x32xf16, #mfma>
      %184 = triton_gpu.convert_layout %183 : (tensor<128x32xf16, #mfma>) -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>
      %185 = triton_gpu.convert_layout %179 : (tensor<32x128xf16, #blocked3>) -> tensor<32x128xf16, #shared2>
      %186 = triton_gpu.convert_layout %185 : (tensor<32x128xf16, #shared2>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>>
      %187 = tt.dot %184, %181, %182 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>> -> tensor<128x128xf32, #mfma>
      %188 = triton_gpu.view_slice %152[0, 96] [128, 32] [1, 1] : tensor<128x128xf16, #mfma> to tensor<128x32xf16, #mfma>
      %189 = triton_gpu.convert_layout %188 : (tensor<128x32xf16, #mfma>) -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>
      %190 = tt.dot %189, %186, %187 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>> -> tensor<128x128xf32, #mfma>
      %191 = "tt.reduce"(%151) <{axis = 1 : i32}> ({
      ^bb0(%arg27: f32, %arg28: f32):
        %195 = arith.addf %arg27, %arg28 : f32
        tt.reduce.return %195 : f32
      }) : (tensor<128x128xf32, #mfma>) -> tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
      %192 = arith.addf %147, %191 : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
      %193 = arith.addi %arg25, %c128_i64 : i64
      %194 = arith.addi %arg26, %c128_i64 : i64
      scf.yield %190, %192, %144, %193, %194 : tensor<128x128xf32, #mfma>, tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>, tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>, i64, i64
    }
    %104 = tt.extern_elementwise %103#1 {libname = "libdevice", libpath = "/triton/python/triton/language/../third_party/hip/lib/bitcode/cuda2gcn.bc", pure = true, symbol = "__nv_log2f"} : (tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>) -> tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
    %105 = arith.addf %103#2, %104 : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
    %106 = triton_gpu.convert_layout %105 : (tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>) -> tensor<128xf32, #blocked1>
    %107 = tt.expand_dims %103#1 {axis = 1 : i32} : (tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>) -> tensor<128x1xf32, #mfma>
    %108 = tt.broadcast %107 : (tensor<128x1xf32, #mfma>) -> tensor<128x128xf32, #mfma>
    %109 = arith.divf %103#0, %108 : tensor<128x128xf32, #mfma>
    %110 = arith.truncf %109 : tensor<128x128xf32, #mfma> to tensor<128x128xf16, #mfma>
    tt.store %53, %106 {cache = 1 : i32, evict = 1 : i32} : tensor<128xf32, #blocked1>
    %111 = tt.broadcast %59 : (tensor<128x1x!tt.ptr<f16, 1>, #mfma>) -> tensor<128x128x!tt.ptr<f16, 1>, #mfma>
    %112 = tt.addptr %111, %99 : tensor<128x128x!tt.ptr<f16, 1>, #mfma>, tensor<128x128xi64, #mfma>
    tt.store %112, %110 {cache = 1 : i32, evict = 1 : i32} : tensor<128x128xf16, #mfma>
    tt.return
  }
}
"""
import tempfile
with tempfile.NamedTemporaryFile(mode='w', suffix='.ttgir') as f:
    f.write(ir) 
    f.flush()
    kernel = triton.compile(f.name)

class _attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale, split_kernel=False):
        # shape constraints
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
        assert Lq == Lk and Lk == Lv
        assert Lk in {16, 32, 64, 128}
        o = torch.empty_like(q, dtype=v.dtype)
        if torch.version.hip is None:
            BLOCK_M = 128
            BLOCK_N = 64 if Lk <= 64 else 32
            num_stages = 4 if Lk <= 64 else 3
            num_warps = 4 if Lk <= 64 else 8
            # Tuning for H100
            if torch.cuda.get_device_capability()[0] == 9:
                num_warps = 8
                num_stages = 7 if Lk >= 64 else 3

        stage = 3 if causal else 1
        grid = lambda META: (
            triton.cdiv(q.shape[2], META['BLOCK_M']),
            q.shape[0] * q.shape[1],
            1
        )

        M = torch.empty((q.shape[0] * q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)



        kernel[(32, 192, 1)](
            q, k, v, sm_scale, M, o,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            o.stride(0), o.stride(1), o.stride(2), 1, 1, q.shape[2]
        )

        ## restore the grid for bwd kernel
        # best_config = _attn_fwd.get_best_config()
        # block_m = int(best_config.__str__().split(",")[0].split("BLOCK_M:")[1])
        # grid = (triton.cdiv(q.shape[2], block_m), q.shape[0] * q.shape[1], 1)

        # ctx.save_for_backward(q, k, v, o, M)
        # ctx.grid = grid
        # ctx.sm_scale = sm_scale
        # ctx.BLOCK_DMODEL = Lk
        # ctx.causal = causal
        # ctx.split_kernel = split_kernel
        return o

    @staticmethod
    def backward(ctx, do):
        # configuration is not supported
        assert(not (ctx.split_kernel and not ctx.causal))
        if torch.version.hip is not None:
            BLOCK = 64
        else:
            BLOCK = 128
        q, k, v, o, L = ctx.saved_tensors
        assert do.is_contiguous()
        assert q.stride() == k.stride() == v.stride() == o.stride() == do.stride()
        do = do.contiguous()
        dq = torch.zeros_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        BATCH, N_HEAD, N_CTX = q.shape[:3]
        delta = torch.empty_like(L)
        do_scaled = torch.empty_like(do)
        # Figure out what BLOCK size fwd used and adjust num_blocks accordingly.
        # If the two are the same, we don't need this but the bwd pass block size
        # is smaller than the fwd so we need this scaling to ensure we loop over all
        # values and don't skip some blocks. 
        # Alternatively we could compute a new grid but this keeps it consistent
        # with fwd and easier to reason about.
        block_scale = (q.shape[2] // ctx.grid[0]) // BLOCK
        _attn_bwd_preprocess[(ctx.grid[0] * ctx.grid[1], )](
            o, do,  #
            do_scaled, delta,  #
            BLOCK_M=block_scale * BLOCK, D_HEAD=ctx.BLOCK_DMODEL,  #
        )
        if not ctx.split_kernel:
            _bwd_kernel[(ctx.grid[1],)](
                q, k, v, ctx.sm_scale,
                o, do_scaled,
                dq, dk, dv,
                L, delta,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                q.shape[0], q.shape[1], q.shape[2],
                block_scale * ctx.grid[0],
                BLOCK_M=BLOCK, BLOCK_N=BLOCK,
                BLOCK_DMODEL=ctx.BLOCK_DMODEL, num_warps=4,
                CAUSAL=ctx.causal,
                num_stages=1,
            )
        else :
            dq = torch.zeros_like(q)
            _bwd_kernel_dk_dv[(block_scale * ctx.grid[0], ctx.grid[1])](
                q, k, v, ctx.sm_scale,
                o, do_scaled,
                dk, dv,
                L, delta,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                q.shape[0], q.shape[1], q.shape[2],
                BLOCK_M=BLOCK, BLOCK_N=BLOCK,
                BLOCK_DMODEL=ctx.BLOCK_DMODEL, num_warps=4,
                num_stages=1,
            )
            _bwd_kernel_dq[ctx.grid](
                q, k, v, ctx.sm_scale,
                o, do_scaled,
                dq,
                L, delta,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                q.shape[0], q.shape[1], q.shape[2],
                BLOCK_M=2*BLOCK, BLOCK_N=BLOCK,
                BLOCK_DMODEL=ctx.BLOCK_DMODEL, num_warps=4, waves_per_eu=1,
                num_stages=1,
            )
        # print(h.asm["ttgir"])
        return dq, dk, dv, None, None, None

attention = _attention.apply


@pytest.mark.parametrize('Z, H, N_CTX, D_HEAD',
                         [(4, 48, 1024, 64),
                          (4, 48, 2048, 64),
                          (4, 48, 4096, 64),
                          (4, 48, 1024, 128),
                          (4, 48, 2048, 128),
                          (4, 48, 4096, 128),
                          #(4, 48, 8192, 64),
                          (4, 48, 16384, 128)
                          ])
@pytest.mark.parametrize('causal', [False, True])
def test_op_fwd(Z, H, N_CTX, D_HEAD, causal, dtype=torch.float16):
    torch.manual_seed(20)
    q = torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0., std=0.5).requires_grad_()
    k = torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0., std=0.5).requires_grad_()
    v = torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0., std=0.5).requires_grad_()
    if TORCH_HAS_FP8:
        q = q.to(torch_dtype)
        k = k.to(torch_dtype)
    sm_scale = 0.5
    dout = torch.randn_like(q, dtype=torch.float16)
    # reference implementation
    M = torch.tril(torch.ones((N_CTX, N_CTX), device="cuda"))
    p = torch.matmul(q.half(), k.transpose(2, 3).half()) * sm_scale
    if causal:
        p[:, :, M == 0] = float("-inf")
    p = torch.softmax(p.float(), dim=-1).half()
    ref_out = torch.matmul(p, v)
    # triton implementation
    tri_out = attention(q, k, v, causal, sm_scale)
    # compare
    torch.testing.assert_close(ref_out, tri_out, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize('Z, H, N_CTX, D_HEAD',
                         [(4, 48, 1024, 64),
                          (4, 48, 2048, 64),
                          (4, 48, 4096, 64),
                          (1, 16, 8192, 64),
                          ])
def test_op_bwd(Z, H, N_CTX, D_HEAD, dtype=torch.float16):
    torch.manual_seed(20)
    causal = True
    q = (torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
    k = (torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
    v = (torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())

    sm_scale = 0.5
    split_kernel = True
    dout = torch.randn_like(q)
    # reference implementation
    M = torch.tril(torch.ones((N_CTX, N_CTX), device="cuda"))
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    if causal:
        p[:, :, M == 0] = float("-inf")
    p = torch.softmax(p.float(), dim=-1).half()
    ref_out = torch.matmul(p, v)
    ref_out.backward(dout)
    ref_dv, v.grad = v.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dq, q.grad = q.grad.clone(), None
    # # triton implementation
    tri_out = attention(q, k, v, causal, sm_scale, split_kernel)
    tri_out.backward(dout)
    tri_dv, v.grad = v.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dq, q.grad = q.grad.clone(), None
    # compare
    torch.testing.assert_close(ref_out, tri_out, atol=1e-2, rtol=0)
    if torch.version.hip is None:
        torch.testing.assert_close(ref_dv, tri_dv, atol=1e-2, rtol=0)
    # The current block size for MI200 series is 64x64. This results in
    # larger differences in float results due to rounding.
    else:
        torch.testing.assert_close(ref_dv, tri_dv, atol=5e-2, rtol=0)
    torch.testing.assert_close(ref_dk, tri_dk, atol=5e-2, rtol=1e-2)
    torch.testing.assert_close(ref_dq, tri_dq, atol=5e-2, rtol=1e-2)


try:
    from flash_attn.flash_attn_interface import \
        flash_attn_qkvpacked_func as flash_attn_func
    HAS_FLASH = True
except BaseException:
    HAS_FLASH = False

# vary seq length for fixed head and batch=4
configs = []
for mode in ['fwd']:
    for D_HEAD in [128]:
        if mode == 'bwd' and D_HEAD == 128:
            continue
        for causal in [False]:
            if mode == 'bwd' and causal == False:
                continue
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
                line_vals=['triton'] + (['flash'] if HAS_FLASH else []),
                line_names=['Triton'] + ([f'Flash-{FLASH_VER}'] if HAS_FLASH else []),
                styles=[('red', '-'), ('blue', '-')],
                ylabel='ms',
                plot_name=f'fused-attention-{mode}-d{D_HEAD}-causal={causal}',
                args={
                    'D_HEAD': D_HEAD,
                    'dtype': torch.float16,
                    'mode': mode,
                    'causal': causal,
                },
            ))


@triton.testing.perf_report(configs)
def bench_flash_attention(BATCH, H, N_CTX, D_HEAD, causal, mode, provider, dtype=torch.float16, device="cuda"):
    assert mode in ["fwd", "bwd"]
    warmup = 25
    rep = 100
    split_kernel = False
    # Bwd pass only supports causal=True right now
    if mode == 'bwd':
        causal = True
        split_kernel = True
    if provider == "triton":
        q = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
        k = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
        v = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
        if mode == "fwd" and TORCH_HAS_FP8:
            q = q.to(torch_dtype)
            k = k.to(torch_dtype)
        sm_scale = 1.3
        fn = lambda: attention(q, k, v, causal, sm_scale, split_kernel)
        if mode == 'bwd':
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    if provider == "flash":
        qkv = torch.randn((BATCH, N_CTX, 3, H, D_HEAD), dtype=dtype, device=device, requires_grad=True)
        fn = lambda: flash_attn_func(qkv, causal=causal)
        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    flops_per_matmul = 2.0 * BATCH * H * N_CTX * N_CTX * D_HEAD
    total_flops = 2 * flops_per_matmul
    if causal:
        total_flops *= 0.5
    if mode == "bwd":
        total_flops *= 2.5  # 2.0(bwd) + 0.5(recompute)
    return total_flops / ms * 1e-9


# only works on post-Ampere GPUs right now
bench_flash_attention.run(save_path=".", print_data=True)
