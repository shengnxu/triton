add_mlir_dialect_library(TritonNvidiaGPUROCMTransforms
  MaterializeLoadStore.cpp
  PlanCTA.cpp
  WSDecomposing.cpp
  WSFeasibilityChecking.cpp
  WSPipeline.cpp
  WSMutex.cpp
  WSMaterialization.cpp
  WSFixupMissingAttrs.cpp
  FenceInsertion.cpp
  RewriteTensorPointer.cpp
  Utility.cpp

  DEPENDS
  TritonNvidiaGPUROCMTransformsIncGen

  LINK_LIBS PUBLIC
  TritonROCMIR
  TritonGPUROCMIR
  TritonGPUROCMTransforms
  TritonNvidiaGPUROCMIR
  MLIRTransformUtils
)
