#ifndef TRITON_CONVERSION_ROCM_TRITONGPU_TO_LLVM_SCAN_OP_H
#define TRITON_CONVERSION_ROCM_TRITONGPU_TO_LLVM_SCAN_OP_H

#include "TritonGPUToLLVMBase.h"

using namespace mlir;
using namespace mlir::triton_rocm;

void populateScanOpToLLVMPatterns(
    TritonGPUROCMToLLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    int numWarps, ModuleAxisInfoAnalysis &axisInfoAnalysis,
    ModuleAllocation &allocation,
    ConvertTritonGPUROCMOpToLLVMPatternBase::IndexCacheInfo &indexCacheInfo,
    PatternBenefit benefit);

#endif
