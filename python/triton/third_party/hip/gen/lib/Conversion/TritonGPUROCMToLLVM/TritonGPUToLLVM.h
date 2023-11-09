#ifndef TRITON_CONVERSION_ROCM_TRITONGPU_TO_LLVM_H
#define TRITON_CONVERSION_ROCM_TRITONGPU_TO_LLVM_H

#include "TritonGPUToLLVMBase.h"

using namespace mlir;
using namespace mlir::triton_rocm;

namespace mlir::triton_rocm {

void populateTritonGPUROCMToLLVMPatterns(
    TritonGPUROCMToLLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    int numWarps, ModuleAxisInfoAnalysis &axisInfoAnalysis,
    ModuleAllocation &allocation,
    ConvertTritonGPUROCMOpToLLVMPatternBase::IndexCacheInfo &indexCacheInfo,
    PatternBenefit benefit);

} // namespace mlir::triton

#endif
