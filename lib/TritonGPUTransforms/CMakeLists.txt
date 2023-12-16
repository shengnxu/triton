add_mlir_dialect_library(TritonGPUTransforms_AMD
  AccelerateAMDMatmul.cpp
  DecomposeConversions.cpp
  OptimizeEpilogue.cpp
  RemoveLayoutConversions.cpp
  ReorderInstructions.cpp
  StreamPipeline.cpp

  DEPENDS
  TritonGPUTransformsIncGen_AMD

  LINK_LIBS PUBLIC
  # MLIRTransforms
  # MLIRTransformUtils
  # TritonAnalysis
  # TritonIR
  # TritonGPUIR
  # TritonNvidiaGPUIR
  # MLIRTransformUtils
)
