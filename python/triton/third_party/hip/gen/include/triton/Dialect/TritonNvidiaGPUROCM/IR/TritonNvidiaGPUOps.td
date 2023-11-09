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

#ifndef TRITONNVIDIAGPUROCM_OPS
#define TRITONNVIDIAGPUROCM_OPS

include "triton/Dialect/TritonNvidiaGPUROCM/IR/TritonNvidiaGPUDialect.td"
include "triton/Dialect/TritonNvidiaGPUROCM/IR/TritonNvidiaGPUTypes.td"
include "triton/Dialect/TritonNvidiaGPUROCM/IR/TritonNvidiaGPUAttrDefs.td"
include "mlir/Dialect/Arith/IR/ArithBase.td"
include "triton/Dialect/TritonROCM/IR/TritonTypes.td"
include "triton/Dialect/TritonROCM/IR/TritonAttrDefs.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td" // Pure
include "mlir/Interfaces/InferTypeOpInterface.td" // SameOperandsAndResultType
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/ViewLikeInterface.td"

def Source1IsSharedEncoding: NativeOpTrait<"Source1IsSharedEncoding">;

def ResultsAreSharedEncodingROCM: NativeOpTrait<"ResultsAreSharedEncodingROCM">;

class TTNGROCM_Op<string mnemonic, list<Trait> traits = []> :
    Op<TritonNvidiaGPUROCM_Dialect, mnemonic, traits>;

// --------------------------------------------------------------------------------------------------
// MBarrier related Ops:
// 1, These mbarrier commands are currently not needed, and not taken into consideration:
//    (1), mbarrier.expect_tx
//    (2), mbarrier.arrive_drop
//    (3), mbarrier.complete_tx
//    (4), mbarrier.inval
//
// 2, The mbarriers is supported to be created in vector, and accessed in seperate via tensor.extract.
//    The mbarriers created in vector will have counters initialized in the same configuration. A
//    typical example to demonstrate this:
//
//    %1 = triton_nvidia_gpu.alloc_mbarrier { count = 1 } : tensor<4x!tt.ptr<i64>>
//    scf.for %iv = %lb to %ub step %step iter_args() -> () {
//      %buffer_id = arith.remi %iv, %c4 : i32
//      %2 = triton_nvidia_gpu.extract_mbarrier %1[%buffer_id] : tensor<4xi64>, i32 -> !tt.ptr<i64>
//      triton_nvidia_gpu.mbarrier_arrive %2 {expectTx = 2048} : !tt.ptr<i64> -> ()
//    }
//    ...
//    scf.for %iv = %lb to %ub step %step iter_args() -> () {
//      %buffer_id = arith.remi %iv, %c4 : i32
//      %2 = triton_nvidia_gpu.extract_mbarrier %1[%buffer_id] : tensor<4xi64>, i32 -> !tt.ptr<i64>
//      triton_nvidia_gpu.mbarrier_wait %2, %c0 : !tt.ptr<i64>, i1 -> ()
//    }

def TTNGROCM_AllocMBarrierOp : TTNGROCM_Op<"alloc_mbarrier", [MemoryEffects<[MemAlloc]>]> {
  let summary = "allocate a vector of mbarriers";

  let description = [{
    Allocate and initialize a vector of mbarriers. The size of the vector is implied in the returned type.
    Each mbarrier is initialized as:
    1, the current phase initialized to 0.
    2, the expected arrival count initialized to 'count'.
    3, the pending arrival count initialized to 'count'.
    4, the tx-count initialized to 0.

    Example:

    case a. when created in vector:
    %1 = triton_nvidia_gpu.alloc_mbarrier { count = 1 } : tensor<4xi64>

    case b. when created in scalar:
    %1 = triton_nvidia_gpu.alloc_mbarrier { count = 1 } : !tt.ptr<i64>

  }];

  let assemblyFormat = [{attr-dict `:` type($result)}];

  let arguments = (ins I32Attr:$count);

  let results = (outs AnyTypeOf<[TTROCM_Ptr, I64Tensor]>:$result);
}

def TTNGROCM_ExtractMBarrierOp : TTNGROCM_Op<"extract_mbarrier", [Pure]> {
  let summary = "extract a mbarrier from a vector of mbarriers";

  let description = [{
    Extract a mbarrier from a vector of mbarriers

    Example:

    %1 = triton_nvidia_gpu.extract_mbarrier %mbarriers[%idx] : tensor<4xi64>, index -> !tt.ptr<i64>

  }];

  let assemblyFormat = "$tensor `[` $index `]` attr-dict `:` type($tensor) `,` type($index) `->` type($result)";

  let arguments = (ins I64Tensor:$tensor, I32:$index);

  let results = (outs TTROCM_Ptr:$result);
}

def TTNGROCM_MBarrierWaitOp : TTNGROCM_Op<"mbarrier_wait", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "mbarrier wait";

  let description = [{
    This operation defining the waiting action for a mbarrier.
    The subsequent operations should not execute until this operation completes waiting.

    Example:

    triton_nvidia_gpu.mbarrier_wait %0, %1 : !tt.ptr<i64>

  }];

  let arguments = (ins TTROCM_Ptr:$mbarrier, I1: $phase);

  let assemblyFormat = "$mbarrier `,` $phase attr-dict `:` type($mbarrier)";
}

def TTNGROCM_MBarrierArriveOp : TTNGROCM_Op<"mbarrier_arrive", [AttrSizedOperandSegments,
                                                      MemoryEffects<[MemWrite]>]> {
  let summary = "mbarrier arrive";

  let description = [{
    This operation defining the arriving action for a mbarrier.
    txCount:
        An optional attribute that set tx-count. This Op will be lowered into
        mbarrier.arrive.expect_tx if the optional attribute exist.
    trackAsyncOp:
        If true, this op will be lowered into cp.async.mbarrier.arrive.noinc.
    pred:
        Only perform arrive action when pred is true.
    remoteCtaId:
        if set, perform an remote arrive action.

    Example:

    triton_nvidia_gpu.mbarrier_arrive %0 {trackAsyncOp = false} : !tt.ptr<i64>

  }];

  let arguments = (ins TTROCM_Ptr:$mbarrier,
                       Optional<I1>:$pred,
                       Optional<I32>:$remoteCtaId,
                       I1Attr: $trackAsyncOp,
                       DefaultValuedAttr<I32Attr, "0">: $txCount
                  );

  let assemblyFormat = "operands attr-dict `:` type(operands)";
}

def TTNGROCM_FenceAsyncSharedOp : TTNGROCM_Op<"fence_async_shared"> {
  let arguments = (ins BoolAttr:$bCluster);

  let summary = "fence proxy async";

  let assemblyFormat = "attr-dict";

  let extraClassDeclaration = [{
    static bool isSupported(int computeCapability) {
      return computeCapability >= 90;
    }
  }];
}

// TODO[goostavz]: ThreadId & ClusterCTAId should not be exposed to
//                 ttgpu level. Remove them when async dialect is ready.
def TTNGROCM_GetThreadIdOp : TTNGROCM_Op<"get_thread_id", [Pure]> {
  let description = [{
    Returns the one dimensional threadId.
  }];

  let results = (outs I32:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}

def TTNGROCM_GetClusterCTAIdOp : TTNGROCM_Op<"get_cluster_cta_id", [Pure]> {
  let description = [{
    Returns the one dimensional cluster_cta_id.
  }];

  let results = (outs I32:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}

def TTNGROCM_NamedBarrierArriveOp : TTNGROCM_Op<"bar_arrive", []> {
  let summary = "named barrier arrive";

  let arguments = (ins I32:$bar, I32: $numThreads);

  let assemblyFormat = "$bar `,` $numThreads attr-dict `:` type(operands)";
}

def TTNGROCM_NamedBarrierWaitOp : TTNGROCM_Op<"bar_wait", []> {
  let summary = "named barrier wait";

  let arguments = (ins I32:$bar, I32: $numThreads);

  let assemblyFormat = "$bar `,` $numThreads attr-dict `:` type(operands)";
}

def TTNGROCM_InsertSliceAsyncV2Op : TTNGROCM_Op<"insert_slice_async_v2",
                                      [AttrSizedOperandSegments,
                                       ResultsAreSharedEncodingROCM,
                                       // TODO: Check if MemWrite will degrade performance of non-warp-specialized kernel
                                       MemoryEffects<[MemRead, MemWrite]>]> {

  let arguments = (ins AnyTypeOf<[TTROCM_Ptr, TTROCM_PtrTensor]>:$src, TTROCM_Tensor:$dst,
                       I32:$index, TTROCM_Ptr:$mbar,
                       Optional<AnyTypeOf<[I1Tensor, I1]>>:$mask, Optional<TTROCM_Type>:$other,
                       TTROCM_CacheModifierAttr:$cache, TTROCM_EvictionPolicyAttr:$evict,
                       BoolAttr:$isVolatile, I32Attr:$axis);

  let results = (outs TTROCM_Tensor:$result);

  let assemblyFormat = "operands attr-dict `:` type(operands) `->` type($result)";
}

// TODO: the abstraction of barriers in ttgpu level is pending, will revisit later
// def TTNG_AwaitOp : TTNG_Op<"await", []> {
//   let arguments = (ins TTNG_TokenType:$token);
//   let assemblyFormat = "$token attr-dict `:` type($token)";
// }

def TTNGROCM_ClusterArriveOp : TTNGROCM_Op<"cluster_arrive", []> {
  let arguments = (ins I1Attr:$relaxed);
  let assemblyFormat = "attr-dict";
}

def TTNGROCM_ClusterWaitOp : TTNGROCM_Op<"cluster_wait", []> {
  let assemblyFormat = "attr-dict";
}

//
// DotAsync Op
//
def TTNGROCM_DotAsyncOp : TTNGROCM_Op<"dot_async", [Pure,
                             DeclareOpInterfaceMethods<InferTypeOpInterface>,
                             TypesMatchWith<"result's type matches accumulator's type",
                                            "d", "c", "$_self">]> {
    let summary = "dot async";

    let description = [{
        $d = matrix_multiply($a, $b) + $c
    }];

    let arguments = (ins TTROCM_FpIntTensor:$a,
                         TTROCM_FpIntTensor:$b,
                         TTROCM_FpIntTensor:$c,
                         BoolAttr:$allowTF32,
                         I32Attr:$maxNumImpreciseAcc);

    let results = (outs TTROCM_FpIntTensor:$d);

    let assemblyFormat = "$a`,` $b`,` $c attr-dict `:` type($a) `*` type($b) `->` type($d)";
}

def TTNGROCM_DotWaitOp : TTNGROCM_Op<"dot_wait", [DeclareOpInterfaceMethods<InferTypeOpInterface>,
                                          AllTypesMatch<["inputs", "outputs"]>]> {
  let summary = "dot wait";
  let arguments = (ins Variadic<TTROCM_FpIntTensor>:$inputs, I32Attr:$pendings);
  let results = (outs Variadic<TTROCM_FpIntTensor>:$outputs);
  let description = [{
    This operation defining the waiting action for a async dot, MMAv3 .e.g.
    The subsequent operations should not execute until this operation completes waiting.
  }];

  let assemblyFormat = "$inputs attr-dict `:` type($inputs)";
}

def TTNGROCM_StoreAsyncOp : TTNGROCM_Op<"store_async",
                              [MemoryEffects<[MemWrite]>]> {
  let summary = "store asynchronous by a tensor pointer";
  let arguments = (ins TTROCM_TensorPtr:$dst, TTROCM_Tensor:$src,
                       DefaultValuedAttr<TTROCM_CacheModifierAttr, "triton_rocm::CacheModifier::NONE">:$cache);
  let assemblyFormat = "operands attr-dict `:` type(operands)";
}

def TTNGROCM_GetAgentIdOp : TTNGROCM_Op<"get_agent_id", [Pure]> {
  let results = (outs I32:$result);

  let builders = [OpBuilder<(ins)>];

  let assemblyFormat = "attr-dict `:` type($result)";
}

//
// Token
//

def TTNGROCM_CreateTokenOp : TTNGROCM_Op<"create_token"> {
  let results = (outs TensorOf<[TTNGROCM_TokenType]>:$result);

  let arguments = (ins I32Attr:$num);

  let builders = [OpBuilder<(ins "uint32_t":$num)>];

  let assemblyFormat = "attr-dict `:` type($result)";
}

def TTNGROCM_ProducerAcquireOp : TTNGROCM_Op<"producer_acquire"> {
  let arguments = (ins TensorOf<[TTNGROCM_TokenType]>:$token, I32:$idx);

  let assemblyFormat = "$token `,` $idx attr-dict `:` type(operands)";
}

def TTNGROCM_ProducerCommitOp : TTNGROCM_Op<"producer_commit"> {
  let arguments = (ins TensorOf<[TTNGROCM_TokenType]>:$token, I32:$idx);

  let assemblyFormat = "$token `,` $idx attr-dict `:` type(operands)";
}

def TTNGROCM_ConsumerWaitOp : TTNGROCM_Op<"consumer_wait"> {
  let arguments = (ins TensorOf<[TTNGROCM_TokenType]>:$token, I32:$idx);

  let assemblyFormat = "$token `,` $idx attr-dict `:` type(operands)";
}

def TTNGROCM_ConsumerReleaseOp : TTNGROCM_Op<"consumer_release"> {
  let arguments = (ins TensorOf<[TTNGROCM_TokenType]>:$token, I32:$idx);

  let assemblyFormat = "$token `,` $idx attr-dict `:` type(operands)";
}

//
// Mutex
//

def TTNGROCM_GetMutexRoleIdOp : TTNGROCM_Op<"get_mutex_role_id"> {
  let results = (outs I32:$result);

  let arguments = (ins I32Attr:$num);

  let builders = [OpBuilder<(ins "uint32_t":$num)>];

  let assemblyFormat = "attr-dict `:` type($result)";
}

def TTNGROCM_CreateMutexOp : TTNGROCM_Op<"create_mutex"> {
  let results = (outs TTNGROCM_MutexType:$result);

  let builders = [OpBuilder<(ins)>];

  let assemblyFormat = "attr-dict `:` type($result)";
}

def TTNGROCM_LockOp : TTNGROCM_Op<"lock"> {
  let arguments = (ins TTNGROCM_MutexType:$mutex);

  let assemblyFormat = "$mutex attr-dict `:` type(operands)";
}

def TTNGROCM_UnlockOp : TTNGROCM_Op<"unlock"> {
  let arguments = (ins TTNGROCM_MutexType:$mutex);

  let assemblyFormat = "$mutex attr-dict `:` type(operands)";
}

def TTNGROCM_RegAllocOp : TTNGROCM_Op<"reg_alloc", []> {
  let summary = "register allocation";

  let arguments = (ins I32Attr: $regCount);

  let assemblyFormat = "$regCount attr-dict";
}

def TTNGROCM_RegDeallocOp : TTNGROCM_Op<"reg_dealloc", []> {
  let summary = "register deallocation";

  let arguments = (ins I32Attr: $regCount);

  let assemblyFormat = "$regCount attr-dict";
}

#endif
