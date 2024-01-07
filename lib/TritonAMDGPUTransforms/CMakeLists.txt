add_mlir_dialect_library(TritonAMDGPUTransforms
  AccelerateAMDMatmul.cpp
  DecomposeConversions.cpp
  OptimizeEpilogue.cpp
  RemoveLayoutConversions.cpp
  ReorderInstructions.cpp
  StreamPipeline.cpp

  DEPENDS
  TritonAMDGPUTransformsIncGen
)
