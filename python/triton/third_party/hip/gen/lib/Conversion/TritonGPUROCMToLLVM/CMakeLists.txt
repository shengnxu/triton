add_library(rocm_libraries SHARED IMPORTED )
set_target_properties(rocm_libraries PROPERTIES IMPORTED_LOCATION ${ROCM_LIBRARIES})
# Separate out PTX/GCN builders to avoid cyclic dependencies as TritonAnalysis
# depends on it.
set(LLVM_OPTIONAL_SOURCES
    GCNAsmFormat.cpp
    PTXAsmFormat.cpp
  )

add_mlir_conversion_library(TritonGPUROCMToLLVM
    ConvertLayoutOpToLLVM/SharedToDotOperandFMA.cpp
    ConvertLayoutOpToLLVM/SharedToDotOperandMMAv1.cpp
    ConvertLayoutOpToLLVM/SharedToDotOperandMMAv2.cpp
    ConvertLayoutOpToLLVM/SharedToDotOperandMFMA.cpp
    ConvertLayoutOpToLLVM.cpp
    DotOpToLLVM/FMA.cpp
    DotOpToLLVM/MMAv1.cpp
    DotOpToLLVM/MMAv2.cpp
    DotOpToLLVM/WGMMA.cpp
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
    ${PROJECT_SOURCE_DIR}/include/triton/Conversion/TritonGPUROCMToLLVM
    ${PROJECT_BINARY_DIR}/include/triton/Conversion/TritonGPUROCMToLLVM

    DEPENDS
    TritonGPUROCMConversionPassIncGen

    LINK_COMPONENTS
    Core

    LINK_LIBS PUBLIC
    GCNASMBuilder
    MLIRIR
    MLIRPass
    MLIRGPUDialect
    MLIRGPUToNVVMTransforms
    MLIRGPUToROCDLTransforms
    MLIRGPUTransforms
    TritonAnalysisROCM
    TritonROCMIR
    TritonGPUROCMIR
    TritonGPUROCMTransforms
    TritonNvidiaGPUROCMTransforms
    NVGPUROCMIR
    rocm_libraries
)

add_mlir_library(GCNASMBuilder
  GCNAsmFormat.cpp
  PTXAsmFormat.cpp

  DEPENDS
  TritonROCMTableGen

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRLLVMDialect
)
