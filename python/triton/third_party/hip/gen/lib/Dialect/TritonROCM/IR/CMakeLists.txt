add_mlir_dialect_library(TritonROCMIR
  Dialect.cpp
  Ops.cpp
  Types.cpp
  Traits.cpp

  DEPENDS
  TritonROCMTableGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRArithDialect
  MLIRMathDialect
  MLIRSCFDialect
)
