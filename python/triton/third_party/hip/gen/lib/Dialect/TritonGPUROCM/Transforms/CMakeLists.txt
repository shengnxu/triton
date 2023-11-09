add_mlir_dialect_library(TritonGPUROCMTransforms
  AccelerateMatmul.cpp
  AccelerateAMDMatmul.cpp
  Coalesce.cpp
  DecomposeConversions.cpp
  OptimizeDotOperands.cpp
  OptimizeEpilogue.cpp
  OptimizeThreadLocality.cpp
  Pipeliner/MatmulLoopPipeline.cpp
  Pipeliner/PipelineExpander.cpp
  Pipeliner/SoftwarePipeliner.cpp
  Prefetch.cpp
  RemoveLayoutConversions.cpp
  ReorderInstructions.cpp
  StreamPipeline.cpp
  TritonGPUConversion.cpp
  Utility.cpp

  DEPENDS
  TritonGPUROCMTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRTransforms
  MLIRTransformUtils
  TritonAnalysisROCM
  TritonROCMIR
  TritonGPUROCMIR
  TritonNvidiaGPUROCMIR
  MLIRTransformUtils
)
