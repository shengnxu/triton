// RUN: (triton-opt %s -split-input-file --convert-rock-to-llvm --mlir-pass-pipeline-crash-reproducer=%t 2>/dev/null; true) | FileCheck %s

#blocked = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 8], order = [0, 1]}>
#mfma = #triton_gpu.mfma<{nonKDim = 32, warpsPerCTA = [2, 4], xdlopsPerWarp = [2, 2]}>
#priv = #gpu.address_space<private>
module attributes {"triton_gpu.num-warps" = 1 : i32} {
  // CHECK-LABEL: convert_memref_to_tensor
  func.func @convert_memref_to_tensor(%arg0: memref<4xvector<16xf32>, 5>) {
    %1 = triton_gpu.memref_to_tensor %arg0 : memref<4xvector<16xf32>, 5> -> tensor<128x256xf32, #mfma>
    return
  }
}
