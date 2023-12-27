add_library(rocm_libraries SHARED IMPORTED )
set(LLVM_OPTIONAL_SOURCES
    GCNAsmFormat.cpp
  )

add_mlir_conversion_library(TritonAMDGPUToLLVM
    ConvertLayoutOpToLLVM/SharedToDotOperandFMA.cpp
    ConvertLayoutOpToLLVM/SharedToDotOperandMFMA.cpp
    ConvertLayoutOpToLLVM.cpp
    DotOpToLLVM/FMA.cpp
    DotOpToLLVM/MFMA.cpp
    DotOpToLLVM.cpp
    ElementwiseOpToLLVM.cpp
    LoadStoreOpToLLVM.cpp
    BarrierOpToLLVM.cpp
    TritonGPUToLLVM.cpp
    GCNAsmFormat.cpp
    PTXAsmFormat.cpp
    TritonGPUToLLVMPass.cpp
    ReduceOpToLLVM.cpp
    ScanOpToLLVM.cpp
    TypeConverter.cpp
    Utility.cpp
    ViewOpToLLVM.cpp
    TensorPtrOpsToLLVM.cpp
    ClusterOpsToLLVM.cpp
    RegReallocOpToLLVM.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/triton/Conversion/TritonAMDGPUToLLVM
    ${PROJECT_BINARY_DIR}/include/triton/Conversion/TritonAMDGPUToLLVM

    DEPENDS
    TritonAMDGPUConversionPassIncGen
)

add_mlir_library(ASMBuilder_AMD
  GCNAsmFormat.cpp

  DEPENDS
  TritonTableGen
)
