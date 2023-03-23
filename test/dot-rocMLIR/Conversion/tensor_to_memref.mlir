// RUN: (triton-opt %s -split-input-file --convert-rock-to-llvm --mlir-pass-pipeline-crash-reproducer=%t 2>/dev/null; true) | FileCheck %s

#blocked = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 8], order = [0, 1]}>
#lds = #triton_gpu.lds<{kpack = 4, order = [1, 0]}>
module attributes {"triton_gpu.num-warps" = 1 : i32} {
  // CHECK: llvm.mlir.global external @global_smem() {addr_space = 3 : i32} : !llvm.array<0 x i8>
  // CHECK-LABEL: convert_tensor_to_memref
  func.func @convert_tensor_to_memref(%arg0: tensor<128x64xf32, #blocked>) {
    %0 = triton_gpu.convert_layout %arg0 : (tensor<128x64xf32, #blocked>) -> tensor<128x64xf32, #lds>
    %1 = triton_gpu.tensor_to_memref %0 : tensor<128x64xf32, #lds> -> memref<128x64xf32, 3>
    return
  }
}
