set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls --name TritonGPUToLLVM_AMD)
add_public_tablegen_target(TritonGPUConversionPassIncGen_AMD)
