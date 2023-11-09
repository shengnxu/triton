#ifndef TRITON_CONVERSION_ROCM_TRITONGPU_TO_LLVM_CONVERT_LAYOUT_OP_H
#define TRITON_CONVERSION_ROCM_TRITONGPU_TO_LLVM_CONVERT_LAYOUT_OP_H

#include "TritonGPUToLLVMBase.h"

using namespace mlir;
using namespace mlir::triton_rocm;

using ::mlir::triton_rocm::gpu_rocm::DotOperandEncodingAttr;

void populateConvertLayoutOpToLLVMPatterns(
    TritonGPUROCMToLLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    int numWarps, ModuleAxisInfoAnalysis &axisInfoAnalysis,
    ModuleAllocation &allocation,
    ConvertTritonGPUROCMOpToLLVMPatternBase::IndexCacheInfo &indexCacheInfo,
    PatternBenefit benefit);

#endif
