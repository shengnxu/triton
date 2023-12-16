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
endif()

set(CMAKE_CXX_STANDARD 17)

# Set optimization flags.
set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -O2 -DUSE_ROCM -D__STDC_FORMAT_MACROS  -fPIC -std=gnu++17 -fvisibility=hidden -fvisibility-inlines-hidden")

# Include the directories that contain additional CMakeLists.txt files.
# This will process the CMakeLists.txt inside these directories.
add_subdirectory(include)
add_subdirectory(lib)

