add_library(rocm_libraries SHARED IMPORTED )
set(LLVM_OPTIONAL_SOURCES
    GCNAsmFormat.cpp
  )

add_mlir_conversion_library(TritonGPUToLLVM_AMD
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
    ${PROJECT_SOURCE_DIR}/include/triton/Conversion/TritonGPUToLLVM
    ${PROJECT_BINARY_DIR}/include/triton/Conversion/TritonGPUToLLVM

    DEPENDS
    TritonGPUConversionPassIncGen
    TritonGPUConversionPassIncGen_AMD

    LINK_COMPONENTS
    Core

    LINK_LIBS PUBLIC
    ASMBuilder
    MLIRIR
    MLIRPass
    MLIRGPUDialect
    MLIRGPUToNVVMTransforms
    MLIRGPUToROCDLTransforms
    MLIRGPUTransforms
    TritonAnalysis
    TritonIR
    TritonGPUIR
    TritonGPUTransforms
    TritonNvidiaGPUTransforms
    NVGPUIR
)

add_mlir_library(ASMBuilder_AMD
  GCNAsmFormat.cpp

  DEPENDS
  TritonTableGen

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRLLVMDialect
)
