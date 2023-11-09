// Copyright (c) 2023 NVIDIA Corporation & Affiliates. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files
// (the "Software"), to deal in the Software without restriction,
// including without limitation the rights to use, copy, modify, merge,
// publish, distribute, sublicense, and/or sell copies of the Software,
// and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
// IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
// CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
// TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
// SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef NVGPUROCM_OPS
#define NVGPUROCM_OPS

include "triton/Dialect/NVGPUROCM/IR/NVGPUDialect.td"
include "triton/Dialect/NVGPUROCM/IR/NVGPUAttrDefs.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td" // SameOperandsAndResultType

def I8Ptr_global : LLVM_IntPtrBase<8, 1>;
def I8Ptr_shared : LLVM_IntPtrBase<8, 3>;
def I64Ptr_shared : LLVM_IntPtrBase<64, 3>;

class NVGPUROCM_Op<string mnemonic, list<Trait> traits = []> :
    LLVM_OpBase<NVGPUROCM_Dialect, mnemonic, traits>;

def NVGPUROCM_WGMMAFenceOp : NVGPUROCM_Op<"wgmma_fence", []> {
  let assemblyFormat = "attr-dict";
}


def NVGPUROCM_WGMMACommitGroupOp : NVGPUROCM_Op<"wgmma_commit_group", []> {
  let assemblyFormat = "attr-dict";
}

def NVGPUROCM_WGMMAWaitGroupOp : NVGPUROCM_Op<"wgmma_wait_group",
[DeclareOpInterfaceMethods<InferTypeOpInterface>,
 AllTypesMatch<["input", "output"]>]> {
  let arguments = (ins LLVM_AnyStruct:$input, I32Attr:$pendings);
  let results = (outs LLVM_AnyStruct:$output);
  let assemblyFormat = "attr-dict";
  let assemblyFormat = "$input attr-dict `:` type($input)";
}

def NVGPUROCM_MBarrierInitOp : NVGPUROCM_Op<"mbarrier_init", [MemoryEffects<[MemWrite]>]> {
  let arguments = (ins I64Ptr_shared:$mbarrier, I1:$pred, I32Attr:$count);
  let assemblyFormat = "$mbarrier `,` $pred attr-dict `:` type($mbarrier)";
}

def MBarrier_ArriveTypeAttr : I32EnumAttr<"MBarriveType",
    "mbarrier arrive type, either 'normal', 'expect_tx', 'cp_async'",
    [
      I32EnumAttrCase<"normal", 0>,
      I32EnumAttrCase<"cp_async", 1>,
      I32EnumAttrCase<"expect_tx", 2>,
      I32EnumAttrCase<"remote", 3>,
    ]>{
  let cppNamespace = "::mlir::triton_rocm::nvgpu";
}

def NVGPUROCM_MBarrierArriveOp : NVGPUROCM_Op<"mbarrier_arrive", []> {
  let arguments = (ins I64Ptr_shared:$mbarrier, I1:$pred, Optional<I32>:$ctaId, MBarrier_ArriveTypeAttr:$arriveType, DefaultValuedAttr<I32Attr, "0">:$txCount);
  let assemblyFormat = "$mbarrier `,` $pred (`,` $ctaId^)? attr-dict `:` type($mbarrier)";
}

def NVGPUROCM_MBarrierWaitOp : NVGPUROCM_Op<"mbarrier_wait", []> {
  let arguments = (ins I64Ptr_shared:$mbarrier, I1:$phase);
  let assemblyFormat = "$mbarrier `,` $phase attr-dict `:` type(operands)";
}

def NVGPUROCM_NamedBarrierArriveOp : NVGPUROCM_Op<"bar_arrive", []> {
  let arguments = (ins I32:$bar, I32:$numThreads);
  let assemblyFormat = "$bar `,` $numThreads attr-dict `:` type(operands)";
}

def NVGPUROCM_NamedBarrierWaitOp : NVGPUROCM_Op<"bar_wait", []> {
  let arguments = (ins I32:$bar, I32:$numThreads);
  let assemblyFormat = "$bar `,` $numThreads attr-dict `:` type(operands)";
}

def WGMMADesc_ModeAttr : I32EnumAttr<"WGMMADescMode",
    "wgmma desc mode, either 'none', 'swizzle128', 'swizzle64', or 'swizzle32'",
    [
      I32EnumAttrCase<"none", 0>,
      I32EnumAttrCase<"swizzle128", 1>,
      I32EnumAttrCase<"swizzle64", 2>,
      I32EnumAttrCase<"swizzle32", 3>
    ]>{
  let cppNamespace = "::mlir::triton_rocm::nvgpu";
}

def NVGPUROCM_WGMMADescCreateOp : NVGPUROCM_Op<"wgmma_desc_create", []> {
  let arguments = (ins LLVM_AnyPointer:$buffer, I32:$height, WGMMADesc_ModeAttr:$mode, I64Attr:$swizzling);
  let builders = [
    OpBuilder<(ins "Value":$buffer,
                     "Value":$height,
                     "WGMMADescMode":$mode), [{
                      uint32_t mode_ = static_cast<uint32_t>(mode);
                      uint64_t swizzling = (mode_ == 1 ? 128 : mode_ == 2 ? 64 : 32);
                      build($_builder, $_state, $_builder.getIntegerType(64), buffer, height, WGMMADescModeAttr::get($_builder.getContext(), mode), $_builder.getI64IntegerAttr(swizzling));
                     }]>];
  let results = (outs I64:$res);
  let assemblyFormat = "$buffer `,` $height attr-dict `:` functional-type(operands, results)";
}

def NVGPUROCM_TMALoadTiledOp : NVGPUROCM_Op<"tma_load_tiled", [AttrSizedOperandSegments]> {
  let arguments = (ins I8Ptr_shared:$dst, I64Ptr_shared:$mbarrier, I8Ptr_global:$tmaDesc, I64:$l2Desc,
                       I1:$pred, Variadic<I32>:$coords, Optional<I16>:$mcastMask);
  let assemblyFormat = "operands attr-dict `:` type(operands)";
}

def NVGPUROCM_TMALoadIm2colOp : NVGPUROCM_Op<"tma_load_im2col", []> {
  let arguments = (ins I8Ptr_shared:$dst, I64Ptr_shared:$mbarrier, I8Ptr_global:$tmaDesc, I64:$l2Desc, LLVM_AnyStruct:$im2colOffsets, I1:$pred, Variadic<I32>:$coords, I16Attr:$mcastMask);
  let assemblyFormat = "operands attr-dict `:` type(operands)";
}

def WGMMA_LayoutAttr : I32EnumAttr<"WGMMALayout",
    "wgmma layout, either 'row' or 'col'",
    [
      I32EnumAttrCase<"row", 0>,
      I32EnumAttrCase<"col", 1>
    ]>{
  let cppNamespace = "::mlir::triton_rocm::nvgpu";
}

def WGMMA_EltTypeAttr : I32EnumAttr<"WGMMAEltType",
    "wgmma operand type, either 's8', 's32', 'e4m3', 'e5m2', 'f16', 'bf16', 'tf32', or 'f32'",
    [
      I32EnumAttrCase<"s8", 0>,
      I32EnumAttrCase<"s32", 1>,
      I32EnumAttrCase<"e4m3", 2>,
      I32EnumAttrCase<"e5m2", 3>,
      I32EnumAttrCase<"f16", 4>,
      I32EnumAttrCase<"bf16", 5>,
      I32EnumAttrCase<"tf32", 6>,
      I32EnumAttrCase<"f32", 7>
    ]>{
  let cppNamespace = "::mlir::triton_rocm::nvgpu";
}

def WGMMA_OperandType : AnyTypeOf<[LLVM_AnyStruct, I64], "wgmma operand A/B type">;

def NVGPUROCM_WGMMAOp : NVGPUROCM_Op<"wgmma", []> {
  let arguments = (ins WGMMA_OperandType:$opA, WGMMA_OperandType:$opB, Optional<LLVM_AnyStruct>:$opC,
                   I32Attr:$m, I32Attr:$n, I32Attr:$k,
                   WGMMA_EltTypeAttr:$eltTypeC, WGMMA_EltTypeAttr:$eltTypeA, WGMMA_EltTypeAttr:$eltTypeB,
                   WGMMA_LayoutAttr:$layoutA, WGMMA_LayoutAttr:$layoutB);
  let results = (outs LLVM_AnyStruct:$res);
  let assemblyFormat = "$opA `,` $opB (`,` $opC^)? attr-dict `:` functional-type(operands, $res)";
}

def NVGPUROCM_CGABarrierSyncOp : NVGPUROCM_Op<"cga_barrier_sync", []> {
  let assemblyFormat = "attr-dict";
}

def NVGPUROCM_CGABarrierArriveOp : NVGPUROCM_Op<"cga_barrier_arrive", []> {
  let assemblyFormat = "attr-dict";
}

def NVGPUROCM_CGABarrierWaitOp : NVGPUROCM_Op<"cga_barrier_wait", []> {
  let assemblyFormat = "attr-dict";
}

def NVGPUROCM_LoadDSmemOp : NVGPUROCM_Op<"load_dsmem", [MemoryEffects<[MemRead]>]> {
  let arguments = (ins LLVM_AnyPointer:$addr, I32:$ctaId, I32Attr:$bitwidth, I32Attr:$vec);
  let builders = [
      OpBuilder<(ins "Type":$resultTy, "Value":$addr, "Value":$ctaId)>,
      OpBuilder<(ins "Value":$addr, "Value":$ctaId, "unsigned":$bitwidth, "unsigned":$vec)>,
      OpBuilder<(ins "Value":$addr, "Value":$ctaId, "unsigned":$bitwidth)>
  ];
  let results = (outs LLVM_LoadableType:$result);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def NVGPUROCM_StoreDSmemOp : NVGPUROCM_Op<"store_dsmem", [MemoryEffects<[MemWrite]>]> {
  let arguments = (ins LLVM_AnyPointer:$addr, I32:$ctaId,
                       Variadic<LLVM_LoadableType>:$values, I1:$pred);
  let builders = [
      OpBuilder<(ins "Value":$addr, "Value":$ctaId, "Value":$value, "Value":$pred)>,
  ];
  let assemblyFormat = "operands attr-dict `:` type(operands)";
  let extraClassDeclaration = [{
      unsigned getBitwidth();
      unsigned getVec();
  }];
}

def NVGPUROCM_FenceAsyncSharedOp : NVGPUROCM_Op<"fence_async_shared", []> {
  let arguments = (ins BoolAttr:$bCluster);
  let assemblyFormat = "attr-dict";
}

def NVGPUROCM_FenceMBarrierInitOp : NVGPUROCM_Op<"fence_mbarrier_init", []> {
  let assemblyFormat = "attr-dict";
}

def NVGPUROCM_ClusterArriveOp : NVGPUROCM_Op<"cluster_arrive", []> {
  let arguments = (ins I1Attr:$relaxed);

  let assemblyFormat = "attr-dict";
}

def NVGPUROCM_ClusterWaitOp : NVGPUROCM_Op<"cluster_wait", []> {
  let assemblyFormat = "attr-dict";
}

def NVGPUROCM_TMAStoreTiledOp : NVGPUROCM_Op<"tma_store_tiled", [MemoryEffects<[MemWrite]>]> {
  let arguments = (ins I8Ptr_global:$tmaDesc, I8Ptr_shared:$src, I1:$pred, Variadic<I32>:$coords);
  let assemblyFormat = "operands attr-dict `:` type(operands)";
}

def NVGPUROCM_StoreMatrixOp : NVGPUROCM_Op<"stmatrix", [MemoryEffects<[MemWrite]>]> {
  let arguments = (ins I8Ptr_shared:$addr, Variadic<I32>:$datas);
  let assemblyFormat = "operands attr-dict `:` type(operands)";
}

def NVGPUROCM_OffsetOfStmatrixV4Op : NVGPUROCM_Op<"offset_of_stmatrix_v4", []> {
  let arguments = (ins I32:$threadId, I32:$rowOfWarp, I32:$elemIdx, I32Attr:$leadingDimOffset, I32Attr:$rowStride, I1Attr:$swizzleEnabled);
  let results = (outs I32:$offset);
  let assemblyFormat = "operands attr-dict `:` type(operands) `->` type($offset)";
}

def NVGPUROCM_OffsetOfSts64Op : NVGPUROCM_Op<"offset_of_sts64", []> {
  let arguments = (ins I32:$threadId, I32:$rowOfWarp, I32:$elemIdx, I32Attr:$leadingDimOffset, I32Attr:$rowStride, I1Attr:$swizzleEnabled);
  let results = (outs I32:$offset);
  let assemblyFormat = "operands attr-dict `:` type(operands) `->` type($offset)";
}

def NVGPUROCM_Sts64Op : NVGPUROCM_Op<"sts64", [MemoryEffects<[MemWrite]>]> {
  let arguments = (ins I32:$offset, AnyTypeOf<[F32, I32]>:$d0, AnyTypeOf<[F32, I32]>:$d1);
  let assemblyFormat = "operands attr-dict `:` type(operands)";
}

def NVGPUROCM_ClusterCTAIdOp : NVGPUROCM_Op<"cluster_id", [Pure]> {
  let results = (outs I32:$result);
  let assemblyFormat = "attr-dict";
}

def NVGPUROCM_RegAllocOp : NVGPUROCM_Op<"reg_alloc", []> {
  let arguments = (ins I32Attr: $regCount);
  let assemblyFormat = "operands attr-dict `:` type(operands)";
}

def NVGPUROCM_RegDeallocOp : NVGPUROCM_Op<"reg_dealloc", []> {
  let arguments = (ins I32Attr: $regCount);
  let assemblyFormat = "operands attr-dict `:` type(operands)";
}

#endif
