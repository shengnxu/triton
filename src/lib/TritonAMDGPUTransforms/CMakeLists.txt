add_mlir_dialect_library(TritonAMDGPUTransforms
  AccelerateAMDMatmul.cpp
  DecomposeConversions.cpp
  OptimizeEpilogue.cpp
  RemoveLayoutConversions.cpp
  ReorderInstructions.cpp
  StreamPipeline.cpp

  DEPENDS
  TritonAMDGPUTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRTransforms
  MLIRTransformUtils
  TritonAnalysis
  TritonIR
  TritonGPUIR
  TritonNvidiaGPUIR
  MLIRTransformUtils
)
