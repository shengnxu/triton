set(LLVM_TARGET_DEFINITIONS Combine.td)
mlir_tablegen(TritonCombine.inc -gen-rewriters)
add_public_tablegen_target(TritonROCMCombineIncGen)

add_mlir_dialect_library(TritonROCMTransforms
  Combine.cpp
  ReorderBroadcast.cpp
  RewriteTensorPointer.cpp

  DEPENDS
  TritonROCMTransformsIncGen
  TritonROCMCombineIncGen

  LINK_LIBS PUBLIC
  MLIRPass
  MLIRTransformUtils
  TritonROCMIR
)
