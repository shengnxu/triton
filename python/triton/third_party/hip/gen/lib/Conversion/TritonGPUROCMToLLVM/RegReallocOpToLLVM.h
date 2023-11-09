#ifndef TRITON_CONVERSION_ROCM_TRITONGPU_TO_LLVM_REGREALLOC_OP_H
#define TRITON_CONVERSION_ROCM_TRITONGPU_TO_LLVM_REGREALLOC_OP_H

#include "TritonGPUToLLVMBase.h"

using namespace mlir;
using namespace mlir::triton_rocm;

void populateRegReallocOpToLLVMPatterns(
    TritonGPUROCMToLLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    int numWarps, ModuleAxisInfoAnalysis &axisInfoAnalysis,
    const ModuleAllocation &allocation, PatternBenefit benefit);

#endif
