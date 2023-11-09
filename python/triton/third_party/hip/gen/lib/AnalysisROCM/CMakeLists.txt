add_mlir_library(TritonAnalysisROCM
  AxisInfo.cpp
  Allocation.cpp
  Membar.cpp
  Alias.cpp
  Utility.cpp

  DEPENDS
  TritonROCMTableGen
  TritonGPUROCMAttrDefsIncGen

  LINK_LIBS PUBLIC
  GCNASMBuilder
  MLIRAnalysis
  MLIRLLVMDialect
  TritonROCMIR
  TritonGPUROCMIR
  TritonNvidiaGPUROCMIR
)
