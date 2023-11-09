#ifndef TRITONGPUROCM_CONVERSION_PASSES_H
#define TRITONGPUROCM_CONVERSION_PASSES_H

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "triton/Conversion/TritonGPUROCMToLLVM/TritonGPUToLLVMPass.h"

namespace mlir {
namespace triton_rocm {

#define GEN_PASS_REGISTRATION
#include "triton/Conversion/TritonGPUROCMToLLVM/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif
