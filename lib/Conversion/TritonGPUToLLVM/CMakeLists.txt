add_mlir_conversion_library(TritonGPUToLLVM
    TritonGPUToRockPass.cpp
    RockToLLVMPass.cpp
    TritonGPUToLLVM.cpp
    GCNAsmFormat.cpp
    PTXAsmFormat.cpp
    ConvertLayoutOpToLLVM/SharedToDotOperandFMA.cpp
    ConvertLayoutOpToLLVM/SharedToDotOperandMMAv1.cpp
    ConvertLayoutOpToLLVM/SharedToDotOperandMMAv2.cpp
    ConvertLayoutOpToLLVM.cpp
    DotOpToLLVM/FMA.cpp
    DotOpToLLVM/MMAv1.cpp
    DotOpToLLVM/MMAv2.cpp
    DotOpToLLVM.cpp
    ElementwiseOpToLLVM.cpp
    LoadStoreOpToLLVM.cpp
    TritonGPUToLLVM.cpp
    GCNAsmFormat.cpp
    PTXAsmFormat.cpp
    ReduceOpToLLVM.cpp
    Utility.cpp
    TypeConverter.cpp
    ViewOpToLLVM.cpp
    TensorMemRefOpToLLVM.cpp
    GpuAllocOpToLLVM.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/triton/Conversion/TritonGPUToLLVM
    ${PROJECT_BINARY_DIR}/include/triton/Conversion/TritonGPUToLLVM

    DEPENDS
    TritonGPUConversionPassIncGen

    LINK_COMPONENTS
    Core

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRGPUOps
    MLIRGPUToNVVMTransforms
    MLIRGPUToROCDLTransforms
    MLIRGPUTransforms
    TritonAnalysis
    TritonIR
    TritonGPUIR
    TritonGPUTransforms
    MLIRRockUtility
)
