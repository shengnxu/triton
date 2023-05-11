#ifndef TRITON_CONVERSION_TRITONTOTRITONGPU_TRITONTOTRITONGPUPASS_H
#define TRITON_CONVERSION_TRITONTOTRITONGPU_TRITONTOTRITONGPUPASS_H

#include <memory>

namespace mlir {

#define GEN_PASS_DECL_CONVERTTRITONTOTRITONGPU
#include "triton/Conversion/TritonToTritonGPU/Passes.h.inc"

class ModuleOp;
template <typename T> class OperationPass;

namespace triton {

constexpr static char AttrNumWarpsName[] = "triton_gpu.num-warps";
constexpr static char AttrKPackName[] = "triton_gpu.kpack";
constexpr static char AttrMPerWaveName[] = "triton_gpu.mPerWave";

// Create the pass with numWarps passed from cl::opt.
std::unique_ptr<OperationPass<ModuleOp>> createConvertTritonToTritonGPUPass();

// Create the pass with numWarps set explicitly.
std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonToTritonGPUPass(int numWarps, int kpack, int mPerWave);

} // namespace triton
} // namespace mlir

#endif
