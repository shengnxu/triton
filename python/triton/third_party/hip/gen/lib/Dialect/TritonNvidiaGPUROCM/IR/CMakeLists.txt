add_mlir_dialect_library(TritonNvidiaGPUROCMIR
  Dialect.cpp
  Ops.cpp
  Traits.cpp
  Types.cpp

  DEPENDS
  TritonNvidiaGPUROCMTableGen
  TritonNvidiaGPUROCMAttrDefsIncGen

  LINK_LIBS PUBLIC
  TritonROCMIR
  TritonGPUROCMIR
)
