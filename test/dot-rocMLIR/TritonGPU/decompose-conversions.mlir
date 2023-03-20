// RUN: triton-opt %s -split-input-file --tritongpu-decompose-conversions 2>&1 | FileCheck %s

#blocked = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [32, 2], warpsPerCTA = [1, 8], order = [0, 1]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 8], order = [0, 1]}>
// CHECK: #lds{{.*}} = #triton_gpu.lds{{.*}}
// CHECK: #lds{{.*}} = #triton_gpu.lds{{.*}}
module attributes {"triton_gpu.num-warps" = 8 : i32} {
  func.func public @kernel_decompose(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: i32 {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}) {
    %cst_a = arith.constant dense<0.000000e+00> : tensor<128x64xf16, #blocked>
    %cst_b = arith.constant dense<0.000000e+00> : tensor<64x256xf16, #blocked1>
    // CHECK: triton_gpu.convert_layout {{.*}} : (tensor<128x64xf16, #blocked>) -> tensor<128x64xf16, [[$lds_layout1:#lds.*]]>
    // CHECK-NEXT: triton_gpu.convert_layout {{.*}} : (tensor<128x64xf16, [[$lds_layout1]]>) -> tensor<128x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #triton_gpu.mfma<{nonKDim = 32, warpsPerCTA = [2, 4], xdlopsPerWarp = [2, 2]}>}>>
    %1 = triton_gpu.convert_layout %cst_a : (tensor<128x64xf16, #blocked>) -> tensor<128x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #triton_gpu.mfma<{nonKDim = 32, warpsPerCTA = [2, 4], xdlopsPerWarp = [2, 2]}>}>>
    // CHECK: triton_gpu.convert_layout {{.*}} : (tensor<64x256xf16, #blocked1>) -> tensor<64x256xf16, [[$lds_layout2:#lds.*]]>
    // CHECK-NEXT: triton_gpu.convert_layout {{.*}} : (tensor<64x256xf16, [[$lds_layout2]]>) -> tensor<64x256xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #triton_gpu.mfma<{nonKDim = 32, warpsPerCTA = [2, 4], xdlopsPerWarp = [2, 2]}>}>>
    %2 = triton_gpu.convert_layout %cst_b : (tensor<64x256xf16, #blocked1>) -> tensor<64x256xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #triton_gpu.mfma<{nonKDim = 32, warpsPerCTA = [2, 4], xdlopsPerWarp = [2, 2]}>}>>
    return
  }
}
