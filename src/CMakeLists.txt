# Set the minimum required version of cmake for the project.
cmake_minimum_required(VERSION 3.10)

# Set your project's name.
project(TritonAMD)

find_package(MLIR REQUIRED CONFIG PATHS ${MLIR_DIR})
include(TableGen) 
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)

# Assuming TRITON_ROOT_DIR is passed to cmake as -DTRITON_ROOT_DIR=<path>.
# Include the Triton project.
if(DEFINED TRITON_ROOT_DIR)
    include_directories(${TRITON_ROOT_DIR})
    add_subdirectory(${TRITON_ROOT_DIR} triton_build)
    include_directories(${CMAKE_BINARY_DIR}/triton_build/include)
    include_directories(${TRITON_ROOT_DIR}/include)
    include_directories(${TRITON_ROOT_DIR}/python)
endif()

set(CMAKE_CXX_STANDARD 17)

# Set optimization flags.
set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -O2 -DUSE_ROCM -D__STDC_FORMAT_MACROS  -fPIC -std=gnu++17 -fvisibility=hidden -fvisibility-inlines-hidden")

# Include the directories that contain additional CMakeLists.txt files.
# This will process the CMakeLists.txt inside these directories.
add_subdirectory(include)
add_subdirectory(lib)


get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
include_directories(~/.triton/pybind11/pybind11-2.11.1/include/)
set(PYTHON_SRC_PATH ${CMAKE_CURRENT_SOURCE_DIR}/python/)
set(PYTHON_SRC ${PYTHON_SRC_PATH}/main.cc ${PYTHON_SRC_PATH}/triton.cc)
find_package(Python3 REQUIRED COMPONENTS Development Interpreter)
include_directories(${Python3_INCLUDE_DIRS})
link_directories(${Python3_LIBRARY_DIRS})
link_libraries(${Python3_LIBRARIES})
add_link_options(${Python3_LINK_OPTIONS})
add_library(triton_amd SHARED ${PYTHON_SRC})
set(TRITON_LIBRARIES
  TritonAnalysis
  TritonTransforms
  TritonNvidiaGPUTransforms
  TritonLLVMIR
  TritonGPUIR
  TritonGPUTransforms
  TritonGPUToLLVM
  TritonGPUTransforms_AMD
  TritonGPUToLLVM_AMD
  ${conversion_libs}
  ${dialect_libs}

  # optimizations
  MLIRBytecodeWriter
  MLIRPass
  MLIRTransforms
  MLIRLLVMDialect
  MLIRSupport
  MLIRTargetLLVMIRExport
  MLIRMathToLLVM
  MLIRNVVMToLLVMIRTranslation
  MLIRROCDLToLLVMIRTranslation
  MLIRIR
)

if(WIN32)
  target_link_libraries(triton_amd PRIVATE ${TRITON_LIBRARIES}
  )
elseif(APPLE)
  target_link_libraries(triton_amd ${TRITON_LIBRARIES}
  )
else()
  target_link_libraries(triton_amd ${TRITON_LIBRARIES})
endif()

set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--exclude-libs,ALL")
# set(CMAKE_SHARED_LIBRARY_SUFFIX ".so")
# set(PYTHON_LDFLAGS "-undefined dynamic_lookup -flto")
target_link_libraries(triton_amd ${PYTHON_LDFLAGS})