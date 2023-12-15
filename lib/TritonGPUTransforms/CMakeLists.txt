add_mlir_dialect_library(TritonGPUTransforms_AMD
  AccelerateMatmul.cpp
  AccelerateAMDMatmul.cpp
  Coalesce.cpp
  DecomposeConversions.cpp
  OptimizeDotOperands.cpp
  OptimizeEpilogue.cpp
  Pipeline.cpp
  Prefetch.cpp
  RemoveLayoutConversions.cpp
  ReorderInstructions.cpp
  StreamPipeline.cpp
  TritonGPUConversion.cpp
  Utility.cpp

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
