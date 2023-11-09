#ifndef TRITON_DIALECT_TRITONROCM_TRANSFORMS_PASSES_H_
#define TRITON_DIALECT_TRITONROCM_TRANSFORMS_PASSES_H_

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace triton_rocm {

std::unique_ptr<Pass> createCombineOpsPass();

std::unique_ptr<Pass> createReorderBroadcastPass();
std::unique_ptr<Pass>
createRewriteTensorPointerPass(int computeCapability = 80,
                                       bool isROCM = false);

} // namespace triton

#define GEN_PASS_REGISTRATION
#include "triton/Dialect/TritonROCM/Transforms/Passes.h.inc"

} // namespace mlir

#endif
