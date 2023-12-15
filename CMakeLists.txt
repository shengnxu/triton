# Set the minimum required version of cmake for the project.
cmake_minimum_required(VERSION 3.10)

# Set your project's name.
project(TritonAMD)

find_package(MLIR REQUIRED CONFIG PATHS ${MLIR_DIR})
include(TableGen) 

# Assuming TRITON_ROOT_DIR is passed to cmake as -DTRITON_ROOT_DIR=<path>.
# Include the Triton project.
if(DEFINED TRITON_ROOT_DIR)
    add_subdirectory(${TRITON_ROOT_DIR} triton_build)
endif()

# Set the C++ standard you want to use (for example, C++14).
set(CMAKE_CXX_STANDARD 14)

# Set optimization flags.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")

# Include the directories that contain additional CMakeLists.txt files.
# This will process the CMakeLists.txt inside these directories.
add_subdirectory(include)
add_subdirectory(lib)

