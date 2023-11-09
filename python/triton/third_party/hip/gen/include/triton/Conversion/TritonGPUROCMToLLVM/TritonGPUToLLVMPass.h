#ifndef TRITON_CONVERSION_ROCM_TRITONGPU_TO_LLVM_PASS_H
#define TRITON_CONVERSION_ROCM_TRITONGPU_TO_LLVM_PASS_H

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Target/PTXROCM/TmaMetadata.h"

#include <memory>

namespace mlir {

class ModuleOp;
template <typename T> class OperationPass;

namespace triton_rocm {

enum Target { NVVM, ROCDL, Default = NVVM };

#define GEN_PASS_DECL
#include "triton/Conversion/TritonGPUROCMToLLVM/Passes.h.inc"

std::unique_ptr<OperationPass<ModuleOp>> createConvertTritonGPUROCMToLLVMPass();
std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonGPUROCMToLLVMPass(int32_t computeCapability, Target target,
                                 mlir::triton_rocm::gpu_rocm::TMAMetadataTy *tmaMetadata);

} // namespace triton

} // namespace mlir

#endif
