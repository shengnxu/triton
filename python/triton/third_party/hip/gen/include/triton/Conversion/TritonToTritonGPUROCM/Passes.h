#ifndef TRITON_CONVERSION_ROCM_PASSES_H
#define TRITON_CONVERSION_ROCM_PASSES_H

#include "triton/Conversion/TritonToTritonGPUROCM/TritonToTritonGPUPass.h"
#include "triton/Target/PTXROCM/TmaMetadata.h"

namespace mlir {
namespace triton_rocm {

#define GEN_PASS_REGISTRATION
#include "triton/Conversion/TritonToTritonGPUROCM/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif
