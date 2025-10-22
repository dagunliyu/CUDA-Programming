cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(CUDATutorial LANGUAGES CXX CUDA VERSION 11.1)

find_package(CUDA 11.1 REQUIRED)

# 需下载安装cuda toolkit，直接搜即可
set(CUDA_PATH ${CUDA_TOOLKIT_ROOT_DIR})

set(CMAKE_CUDA_COMPILER "D:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\bin\\nvcc.exe")

# set compiler flags
set(CMAKE_C_FLAGS    "${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}  -Xcompiler -Wall")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}  \
                        -gencode=arch=compute_35,code=\\\"sm_35,compute_35\\\" \
                        -gencode=arch=compute_70,code=\\\"sm_70,compute_70\\\" \
                        -gencode=arch=compute_75,code=\\\"sm_75,compute_75\\\" \
                        -gencode=arch=compute_80,code=\\\"sm_80,compute_80\\\" \
                        -gencode=arch=compute_86,code=\\\"sm_86,compute_86\\\" \
                        ")

message("CMAKE_CUDA_FLAGS = ${CMAKE_CUDA_FLAGS}")
message("CUDA_PATH = ${CUDA_PATH}")
message("CMAKE_CUDA_COMPILER = ${CMAKE_CUDA_COMPILER}")

set(CMAKE_CUDA_ARCHITECTURES 35 70 75 80 86)

#输出的可执行文件保存地址
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
    
set(COMMON_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/15_gemv
    ${CUDA_PATH}/include
)

set(COMMON_LIB_DIRS
    ${CUDA_PATH}/lib64
)

#指定头文件路径
include_directories(
    ${COMMON_HEADER_DIRS}
)
#指定待链接的lib路径
link_directories(
    ${COMMON_LIB_DIRS}
)

################1
add_executable(3-add3if
    add3if.cu
)
target_link_libraries(
    3-add3if PUBLIC
    -lcudart
    -lcudadevrt)
