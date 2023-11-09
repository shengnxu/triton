add_mlir_conversion_library(NVGPUROCMToLLVM
    NVGPUToLLVMPass.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/triton/Conversion/NVGPUROCMToLLVM
    ${PROJECT_BINARY_DIR}/include/triton/Conversion/NVGPUROCMToLLVM

    DEPENDS
    NVGPUROCMConversionPassIncGen

    LINK_COMPONENTS
    Core

    LINK_LIBS PUBLIC
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
)
