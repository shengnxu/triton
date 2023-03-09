// Sanity test to ensure every step of the lowering process gets valid MLIR,
// LLVM IR

// RUN: triton-opt -rock-blockwise-gemm-to-threadwise %s
// RUN: triton-opt -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering %s
// RUN: triton-opt -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering -rock-sugar-to-loops %s
// RUN: triton-opt -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering -rock-sugar-to-loops -rock-clean-math %s
// RUN: triton-opt -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering -rock-sugar-to-loops -rock-clean-math -rock-buffer-load-merge %s
// RUN: triton-opt -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering -rock-sugar-to-loops -rock-clean-math -rock-buffer-load-merge -rock-loops-to-cf %s


#blocked = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 2], warpsPerCTA = [1, 8], order = [0, 1]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked2 = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 8], order = [0, 1]}>
#lds = #triton_gpu.lds<{kpack = 4, order = [1, 0]}>
#lds1 = #triton_gpu.lds<{kpack = 4, order = [0, 1]}>
#xdlops_gemm_params = #rock.xdlops_gemm_params<kPerBlock = 16, mPerBlock = 128, nPerBlock = 256, kpack = 4, mPerWave = 64, nPerWave = 64, forceUnroll = true>
module attributes {"triton_gpu.num-warps" = 8 : i32} {
  func.func public @kernel_0d1c2d3d4c5d6d7d8c9d10d11c(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: i32 {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #triton_gpu.mfma<{nonKDim = 32, warpsPerCTA = [2, 4], xdlopsPerWarp = [2, 2]}>>
    %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %1 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %2 = tt.expand_dims %0 {axis = 1 : i32} : (tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>) -> tensor<128x1xi32, #blocked>
    %3 = tt.expand_dims %1 {axis = 1 : i32} : (tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>) -> tensor<128x1xi32, #blocked1>
    %4 = tt.splat %arg0 : (!tt.ptr<f16>) -> tensor<128x1x!tt.ptr<f16>, #blocked>
    %5 = tt.addptr %4, %2 : tensor<128x1x!tt.ptr<f16>, #blocked>, tensor<128x1xi32, #blocked>
    %6 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %7 = tt.expand_dims %6 {axis = 0 : i32} : (tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>) -> tensor<1x64xi32, #blocked>
    %8 = tt.splat %arg1 : (i32) -> tensor<1x64xi32, #blocked>
    %9 = arith.muli %7, %8 : tensor<1x64xi32, #blocked>
    %10 = tt.broadcast %5 : (tensor<128x1x!tt.ptr<f16>, #blocked>) -> tensor<128x64x!tt.ptr<f16>, #blocked>
    %11 = tt.broadcast %9 : (tensor<1x64xi32, #blocked>) -> tensor<128x64xi32, #blocked>
    %12 = tt.addptr %10, %11 : tensor<128x64x!tt.ptr<f16>, #blocked>, tensor<128x64xi32, #blocked>
    %13 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %14 = tt.expand_dims %13 {axis = 1 : i32} : (tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>) -> tensor<64x1xi32, #blocked2>
    %15 = tt.splat %arg2 : (!tt.ptr<f16>) -> tensor<64x1x!tt.ptr<f16>, #blocked2>
    %16 = tt.addptr %15, %14 : tensor<64x1x!tt.ptr<f16>, #blocked2>, tensor<64x1xi32, #blocked2>
    %17 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
    %18 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %19 = tt.expand_dims %17 {axis = 0 : i32} : (tensor<256xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>) -> tensor<1x256xi32, #blocked2>
    %20 = tt.expand_dims %18 {axis = 0 : i32} : (tensor<256xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>) -> tensor<1x256xi32, #blocked1>
    %21 = tt.splat %arg3 : (i32) -> tensor<1x256xi32, #blocked2>
    %22 = arith.muli %19, %21 : tensor<1x256xi32, #blocked2>
    %23 = tt.broadcast %16 : (tensor<64x1x!tt.ptr<f16>, #blocked2>) -> tensor<64x256x!tt.ptr<f16>, #blocked2>
    %24 = tt.broadcast %22 : (tensor<1x256xi32, #blocked2>) -> tensor<64x256xi32, #blocked2>
    %25 = tt.addptr %23, %24 : tensor<64x256x!tt.ptr<f16>, #blocked2>, tensor<64x256xi32, #blocked2>
    %26 = tt.splat %arg7 : (i32) -> tensor<128x1xi32, #blocked1>
    %27 = arith.muli %3, %26 : tensor<128x1xi32, #blocked1>
    %28 = tt.splat %arg6 : (!tt.ptr<f16>) -> tensor<128x1x!tt.ptr<f16>, #blocked1>
    %29 = tt.addptr %28, %27 : tensor<128x1x!tt.ptr<f16>, #blocked1>, tensor<128x1xi32, #blocked1>
    %30 = tt.broadcast %29 : (tensor<128x1x!tt.ptr<f16>, #blocked1>) -> tensor<128x256x!tt.ptr<f16>, #blocked1>
    %31 = tt.broadcast %20 : (tensor<1x256xi32, #blocked1>) -> tensor<128x256xi32, #blocked1>
    %32 = tt.addptr %30, %31 : tensor<128x256x!tt.ptr<f16>, #blocked1>, tensor<128x256xi32, #blocked1>
    %33 = tt.load %12 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x64xf16, #blocked>
    %34 = tt.load %25 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<64x256xf16, #blocked2>
    %35 = triton_gpu.convert_layout %33 : (tensor<128x64xf16, #blocked>) -> tensor<128x64xf16, #lds>
    %36 = triton_gpu.convert_layout %34 : (tensor<64x256xf16, #blocked2>) -> tensor<64x256xf16, #lds1>
    %37 = triton_gpu.tensor_to_memref %35 : tensor<128x64xf16, #lds> -> memref<8192xf16, #gpu.address_space<workgroup>>
    %38 = triton_gpu.tensor_to_memref %36 : tensor<64x256xf16, #lds1> -> memref<16384xf16, #gpu.address_space<workgroup>>
    %c64 = arith.constant 64 : index
    %c64_0 = arith.constant 64 : index
    %c64_1 = arith.constant 64 : index
    %c4 = arith.constant 4 : index
    %39 = rock.workgroup_id : index
    %40 = rock.workitem_id : index
    %41 = arith.divui %40, %c64 : index
    %42 = arith.divui %41, %c4 : index
    %43 = arith.remui %41, %c4 : index
    %44 = arith.muli %42, %c64_0 : index
    %45 = arith.muli %43, %c64_1 : index
    %46 = rock.alloc() : memref<16xvector<4xf16>, #gpu.address_space<private>>
    %47 = rock.alloc() : memref<16xvector<4xf16>, #gpu.address_space<private>>
    %48 = rock.alloc() : memref<4xvector<16xf32>, #gpu.address_space<private>>
    %cst_2 = arith.constant dense<0.000000e+00> : vector<16xf32>
    rock.fill(%48, %cst_2) : memref<4xvector<16xf32>, #gpu.address_space<private>>, vector<16xf32>
    rock.blockwise_gemm_v2 %48 += %46 from %37[%44] * %47 from %38[%45] {blockSize = 512 : i32, ldsBufferOffsetA = 0 : index, ldsBufferOffsetB = 0 : index, params = #xdlops_gemm_params} : memref<4xvector<16xf32>, #gpu.address_space<private>> += memref<16xvector<4xf16>, #gpu.address_space<private>> from memref<8192xf16, #gpu.address_space<workgroup>> * memref<16xvector<4xf16>, #gpu.address_space<private>> from memref<16384xf16, #gpu.address_space<workgroup>>
    %49 = triton_gpu.memref_to_tensor %48 : memref<4xvector<16xf32>, #gpu.address_space<private>> -> tensor<128x256xf32, #triton_gpu.mfma<{nonKDim = 32, warpsPerCTA = [2, 4], xdlopsPerWarp = [2, 2]}>>
    %50 = triton_gpu.convert_layout %49 : (tensor<128x256xf32, #triton_gpu.mfma<{nonKDim = 32, warpsPerCTA = [2, 4], xdlopsPerWarp = [2, 2]}>>) -> tensor<128x256xf32, #blocked1>
    %51 = arith.truncf %50 : tensor<128x256xf32, #blocked1> to tensor<128x256xf16, #blocked1>
    tt.store %32, %51 {cache = 1 : i32, evict = 1 : i32} : tensor<128x256xf16, #blocked1>
    return
  }
}
