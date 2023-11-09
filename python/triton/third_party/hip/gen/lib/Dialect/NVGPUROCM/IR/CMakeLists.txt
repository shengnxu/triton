add_mlir_dialect_library(NVGPUROCMIR
  Dialect.cpp

  DEPENDS
  NVGPUROCMTableGen
  NVGPUROCMAttrDefsIncGen

  LINK_LIBS PUBLIC
  MLIRLLVMDialect
)
