#ifndef TRITON_CONVERSION_ROCM_NVGPU_TO_LLVM_PASS_H
#define TRITON_CONVERSION_ROCM_NVGPU_TO_LLVM_PASS_H

#include <memory>

namespace mlir {

class ModuleOp;
template <typename T> class OperationPass;

namespace triton_rocm {

std::unique_ptr<OperationPass<ModuleOp>> createConvertNVGPUROCMToLLVMPass();

} // namespace triton

} // namespace mlir

#endif
