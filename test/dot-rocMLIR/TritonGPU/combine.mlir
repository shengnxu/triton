// RUN: triton-opt %s -split-input-file  -tritongpu-remove-layout-conversions -tritongpu-accelerate-matmul  -tritongpu-remove-layout-conversions 2>&1 | FileCheck %s

#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [64, 1], warpsPerCTA = [2, 4], order = [0, 1]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [8], order = [0]}>
#blocked2 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [64, 1], warpsPerCTA = [8, 1], order = [0, 1]}>
#blocked3 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [1, 8], order = [0, 1]}>
#blocked4 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [64, 1], warpsPerCTA = [1, 8], order = [0, 1]}>
#blocked5 = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 8], order = [0, 1]}>
#blocked6 = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>
#blocked7 = #triton_gpu.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 64], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked8 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 32], warpsPerCTA = [8, 1], order = [1, 0]}>
module attributes {"triton_gpu.num-warps" = 8 : i32} {
  func.func public @kernel_intro_mfma(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: i32 {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}) {
    // CHECK: arith.constant dense<0.000000e+00> : tensor<128x256xf32, [[$mfmaEnc:#triton_gpu.mfma<.*>]]>
    %cst = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #blocked>
    %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked1>
    %1 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked1>
    %2 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked1>
    %3 = triton_gpu.convert_layout %0 : (tensor<128xi32, #blocked1>) -> tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %4 = tt.expand_dims %3 {axis = 1 : i32} : (tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>) -> tensor<128x1xi32, #blocked2>
    %5 = triton_gpu.convert_layout %4 : (tensor<128x1xi32, #blocked2>) -> tensor<128x1xi32, #blocked>
    %6 = tt.splat %arg0 : (!tt.ptr<f16>) -> tensor<128x1x!tt.ptr<f16>, #blocked>
    %7 = tt.addptr %6, %5 : tensor<128x1x!tt.ptr<f16>, #blocked>, tensor<128x1xi32, #blocked>
    %8 = triton_gpu.convert_layout %2 : (tensor<64xi32, #blocked1>) -> tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked3}>>
    %9 = tt.expand_dims %8 {axis = 0 : i32} : (tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked3}>>) -> tensor<1x64xi32, #blocked3>
    %10 = tt.splat %arg1 : (i32) -> tensor<1x64xi32, #blocked3>
    %11 = arith.muli %9, %10 : tensor<1x64xi32, #blocked3>
    %12 = tt.broadcast %7 : (tensor<128x1x!tt.ptr<f16>, #blocked>) -> tensor<128x64x!tt.ptr<f16>, #blocked>
    %13 = tt.broadcast %11 : (tensor<1x64xi32, #blocked3>) -> tensor<128x64xi32, #blocked3>
    %14 = triton_gpu.convert_layout %13 : (tensor<128x64xi32, #blocked3>) -> tensor<128x64xi32, #blocked>
    %15 = tt.addptr %12, %14 : tensor<128x64x!tt.ptr<f16>, #blocked>, tensor<128x64xi32, #blocked>
    %16 = triton_gpu.convert_layout %2 : (tensor<64xi32, #blocked1>) -> tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %17 = tt.expand_dims %16 {axis = 1 : i32} : (tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>) -> tensor<64x1xi32, #blocked2>
    %18 = triton_gpu.convert_layout %17 : (tensor<64x1xi32, #blocked2>) -> tensor<64x1xi32, #blocked4>
    %19 = tt.splat %arg2 : (!tt.ptr<f16>) -> tensor<64x1x!tt.ptr<f16>, #blocked4>
    %20 = tt.addptr %19, %18 : tensor<64x1x!tt.ptr<f16>, #blocked4>, tensor<64x1xi32, #blocked4>
    %21 = triton_gpu.convert_layout %1 : (tensor<256xi32, #blocked1>) -> tensor<256xi32, #triton_gpu.slice<{dim = 0, parent = #blocked3}>>
    %22 = tt.expand_dims %21 {axis = 0 : i32} : (tensor<256xi32, #triton_gpu.slice<{dim = 0, parent = #blocked3}>>) -> tensor<1x256xi32, #blocked3>
    %23 = tt.splat %arg3 : (i32) -> tensor<1x256xi32, #blocked3>
    %24 = arith.muli %22, %23 : tensor<1x256xi32, #blocked3>
    %25 = tt.broadcast %20 : (tensor<64x1x!tt.ptr<f16>, #blocked4>) -> tensor<64x256x!tt.ptr<f16>, #blocked4>
    %26 = tt.broadcast %24 : (tensor<1x256xi32, #blocked3>) -> tensor<64x256xi32, #blocked3>
    %27 = triton_gpu.convert_layout %26 : (tensor<64x256xi32, #blocked3>) -> tensor<64x256xi32, #blocked4>
    %28 = tt.addptr %25, %27 : tensor<64x256x!tt.ptr<f16>, #blocked4>, tensor<64x256xi32, #blocked4>
    %29 = tt.splat %arg7 : (i32) -> tensor<128x1xi32, #blocked>
    %30 = arith.muli %5, %29 : tensor<128x1xi32, #blocked>
    %31 = tt.splat %arg6 : (!tt.ptr<f16>) -> tensor<128x1x!tt.ptr<f16>, #blocked>
    %32 = tt.addptr %31, %30 : tensor<128x1x!tt.ptr<f16>, #blocked>, tensor<128x1xi32, #blocked>
    %33 = tt.broadcast %32 : (tensor<128x1x!tt.ptr<f16>, #blocked>) -> tensor<128x256x!tt.ptr<f16>, #blocked>
    %34 = tt.broadcast %22 : (tensor<1x256xi32, #blocked3>) -> tensor<128x256xi32, #blocked3>
    %35 = triton_gpu.convert_layout %34 : (tensor<128x256xi32, #blocked3>) -> tensor<128x256xi32, #blocked>
    %36 = tt.addptr %33, %35 : tensor<128x256x!tt.ptr<f16>, #blocked>, tensor<128x256xi32, #blocked>
    %37 = triton_gpu.convert_layout %15 : (tensor<128x64x!tt.ptr<f16>, #blocked>) -> tensor<128x64x!tt.ptr<f16>, #blocked5>
    %38 = tt.load %37 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x64xf16, #blocked5>
    %39 = triton_gpu.convert_layout %38 : (tensor<128x64xf16, #blocked5>) -> tensor<128x64xf16, #blocked>
    %40 = triton_gpu.convert_layout %28 : (tensor<64x256x!tt.ptr<f16>, #blocked4>) -> tensor<64x256x!tt.ptr<f16>, #blocked6>
    %41 = tt.load %40 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<64x256xf16, #blocked6>
    %42 = triton_gpu.convert_layout %41 : (tensor<64x256xf16, #blocked6>) -> tensor<64x256xf16, #blocked4>
    %43 = triton_gpu.convert_layout %39 : (tensor<128x64xf16, #blocked>) -> tensor<128x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked7}>>
    %44 = triton_gpu.convert_layout %42 : (tensor<64x256xf16, #blocked4>) -> tensor<64x256xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked7}>>
    %45 = triton_gpu.convert_layout %cst : (tensor<128x256xf32, #blocked>) -> tensor<128x256xf32, #blocked7>
    // CHECK: tt.dot {{.*}}, {{.*}}, {{.*}} {allowTF32 = true} : tensor<128x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = [[$mfmaEnc]]}>> * tensor<64x256xf16, #triton_gpu.dot_op<{opIdx = 1, parent = [[$mfmaEnc]]}>> -> tensor<128x256xf32, [[$mfmaEnc]]>
    %46 = tt.dot %43, %44, %45 {allowTF32 = true} : tensor<128x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked7}>> * tensor<64x256xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked7}>> -> tensor<128x256xf32, #blocked7>
    // CHECK-NEXT: triton_gpu.convert_layout {{.*}} : (tensor<128x256xf32, [[$mfmaEnc]]>) -> tensor<128x256xf32, #blocked{{.*}}>
    %47 = triton_gpu.convert_layout %46 : (tensor<128x256xf32, #blocked7>) -> tensor<128x256xf32, #blocked>
    %48 = arith.truncf %47 : tensor<128x256xf32, #blocked> to tensor<128x256xf16, #blocked>
    %49 = triton_gpu.convert_layout %36 : (tensor<128x256x!tt.ptr<f16>, #blocked>) -> tensor<128x256x!tt.ptr<f16>, #blocked8>
    %50 = triton_gpu.convert_layout %48 : (tensor<128x256xf16, #blocked>) -> tensor<128x256xf16, #blocked8>
    tt.store %49, %50 {cache = 1 : i32, evict = 1 : i32} : tensor<128x256xf16, #blocked8>
    return
  }
}
