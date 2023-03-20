// RUN: (triton-opt %s -split-input-file --convert-rock-to-llvm --mlir-pass-pipeline-crash-reproducer=%t 2>/dev/null; true) | FileCheck %s

#blocked = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 8], order = [0, 1]}>
#lds = #triton_gpu.lds<{kpack = 4, order = [1, 0]}>
module attributes {"triton_gpu.num-warps" = 1 : i32} {
  // CHECK: llvm.mlir.global external @global_smem() {addr_space = 3 : i32} : !llvm.array<0 x i8>
  // CHECK-LABEL: convert_layout_blocked_lds_diff_order
  func.func @convert_layout_blocked_lds_diff_order(%arg0: tensor<128x64xf32, #blocked>) {
    // CHECK: llvm.store
    // CHECK-SAME: !llvm.ptr<vector<1xf32>, 3>
    // CHECK: llvm.store
    // CHECK-SAME: !llvm.ptr<vector<1xf32>, 3>
    %0 = triton_gpu.convert_layout %arg0 : (tensor<128x64xf32, #blocked>) -> tensor<128x64xf32, #lds>
    return
  }
}

// -----

#blocked1 = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>
#lds1 = #triton_gpu.lds<{kpack = 4, order = [0, 1]}>
module attributes {"triton_gpu.num-warps" = 1 : i32} {
  // CHECK: llvm.mlir.global external @global_smem() {addr_space = 3 : i32} : !llvm.array<0 x i8>
  // CHECK-LABEL: convert_layout_blocked_lds_same_order
  func.func @convert_layout_blocked_lds_same_order(%arg1: tensor<64x256xf32, #blocked1>) {
    // CHECK: llvm.store
    // CHECK-SAME: !llvm.ptr<vector<4xf32>, 3>
    // CHECK: llvm.store
    // CHECK-SAME: !llvm.ptr<vector<4xf32>, 3>
    %1 = triton_gpu.convert_layout %arg1 : (tensor<64x256xf32, #blocked1>) -> tensor<64x256xf32, #lds1>
    return
  }
}
