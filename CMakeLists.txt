include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
add_compile_definitions(USE_ROCM=1)
add_subdirectory(include)
add_subdirectory(lib)

add_library(triton_amd SHARED ${CMAKE_CURRENT_SOURCE_DIR}/python/triton_amd.cc)
set(TRITON_AMD_LIBRARIES
  TritonAMDGPUToLLVM
  TritonAMDGPUTransforms
  ${LLVM_LIBRARY_DIR}/libMLIRNVVMToLLVMIRTranslation.a
  ${LLVM_LIBRARY_DIR}/libMLIRGPUToNVVMTransforms.a
  ${LLVM_LIBRARY_DIR}/libMLIRGPUToROCDLTransforms.a
  ${LLVM_LIBRARY_DIR}/libMLIRGPUToGPURuntimeTransforms.a
  ${LLVM_LIBRARY_DIR}/libMLIRROCDLToLLVMIRTranslation.a
  ${LLVM_LIBRARY_DIR}/libMLIRGPUTransforms.a
  ${LLVM_LIBRARY_DIR}/libMLIRAMDGPUToROCDL.a
  ${LLVM_LIBRARY_DIR}/libMLIRAMDGPUUtils.a
  ${LLVM_LIBRARY_DIR}/libMLIRAMDGPUTransforms.a
  MLIRMemRefToLLVM
  MLIRArithTransforms
  MLIRVectorDialect
  MLIRVectorToLLVM
  MLIRFuncToLLVM
  MLIRROCDLDialect
  MLIRAMDGPUDialect
  MLIRMathToLLVM
  # TODO: shouldn't be there
  TritonNvidiaGPUTransforms
  NVGPUIR
)
target_link_libraries(triton_amd PRIVATE ${TRITON_AMD_LIBRARIES} "-Wl,--no-as-needed" triton)
target_link_libraries(triton_amd ${PYTHON_LDFLAGS})