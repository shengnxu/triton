#ifndef TRITON_TARGET_PTXROCMTRANSLATION_H
#define TRITON_TARGET_PTXROCMTRANSLATION_H

#include <string>

namespace llvm {
class Module;
} // namespace llvm

namespace triton_rocm {

// Translate TritonGPU IR to PTX code.
std::string translateLLVMIRToPTX(llvm::Module &module, int cc, int version,
                                 bool enable_fp_fusion);

} // namespace triton

#endif
