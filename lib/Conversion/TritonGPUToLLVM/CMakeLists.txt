add_mlir_conversion_library(TritonGPUToLLVM
    TritonGPUToRockPass.cpp
    RockToLLVMPass.cpp
    ConvertLayoutOpToLLVM.cpp
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
    DotOpHelpers.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/triton/Conversion/TritonGPUToLLVM

    DEPENDS
    TritonConversionPassIncGen

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
