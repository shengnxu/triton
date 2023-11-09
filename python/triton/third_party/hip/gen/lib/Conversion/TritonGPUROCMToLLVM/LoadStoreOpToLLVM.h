#ifndef TRITON_CONVERSION_ROCM_TRITONGPU_TO_LLVM_LOAD_STORE_OP_H
#define TRITON_CONVERSION_ROCM_TRITONGPU_TO_LLVM_LOAD_STORE_OP_H

#include "TritonGPUToLLVMBase.h"

using namespace mlir;
using namespace mlir::triton_rocm;

void populateLoadStoreOpToLLVMPatterns(
    TritonGPUROCMToLLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    int numWarps, ModuleAxisInfoAnalysis &axisInfoAnalysis,
    ModuleAllocation &allocation,
    ConvertTritonGPUROCMOpToLLVMPatternBase::IndexCacheInfo &indexCacheInfo,
    mlir::triton_rocm::gpu_rocm::TMAMetadataTy *tmaMetadata,
    const TensorPtrMapT *tensorPtrMap, PatternBenefit benefit);

#endif
