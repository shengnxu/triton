#ifndef NVGPU_CONVERSION_PASSES_H
#define NVGPU_CONVERSION_PASSES_H

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "triton/Conversion/NVGPUROCMToLLVM/NVGPUToLLVMPass.h"

namespace mlir {
namespace triton_rocm {

#define GEN_PASS_REGISTRATION
#include "triton/Conversion/NVGPUROCMToLLVM/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif
