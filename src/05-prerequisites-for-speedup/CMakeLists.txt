cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(CUDATutorial LANGUAGES CXX CUDA)

find_package(CUDA 10.0 REQUIRED)
# 需下载安装cuda toolkit，直接搜即可
set(CUDA_PATH ${CUDA_TOOLKIT_ROOT_DIR})

# set compiler flags
set(CMAKE_C_FLAGS    "${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}  -Xcompiler -Wall")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}  \
                        -gencode=arch=compute_70,code=\\\"sm_70,compute_70\\\" \
                        -gencode=arch=compute_75,code=\\\"sm_75,compute_75\\\" \
                        -gencode=arch=compute_80,code=\\\"sm_80,compute_80\\\" \
                        -gencode=arch=compute_86,code=\\\"sm_86,compute_86\\\" \
                        ")
set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86)
#输出的可执行文件保存地址
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
    
add_definitions(-DUSE_DP)

set(COMMON_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/15_gemv
    ${CUDA_PATH}/include
)

set(COMMON_LIB_DIRS
    ${CUDA_PATH}/lib64
)

###############################

add_executable(5-add1cpu
    add1cpu.cu
)
target_include_directories(
    5-add1cpu PRIVATE
    ${CMAKE_SOURCE_DIR}
)
target_link_libraries(
    5-add1cpu PUBLIC
    -lcudart
    -lcudadevrt)

###############################

add_executable(5-add2gpu
    add2gpu.cu
)
target_include_directories(
    5-add2gpu PRIVATE
    ${CMAKE_SOURCE_DIR}
)
target_link_libraries(
    5-add2gpu PUBLIC
    -lcudart
    -lcudadevrt)

###############################

add_executable(5-add3memcpy
    add3memcpy.cu
)
target_include_directories(
    5-add3memcpy PRIVATE
    ${CMAKE_SOURCE_DIR}
)
target_link_libraries(
    5-add3memcpy PUBLIC
    -lcudart
    -lcudadevrt)

#set_property(TARGET 5-add3memcpy APPEND PROPERTY VS_PROJECT_IMPORT "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\extras\\visual_studio_integration\\MSBuildExtensions\\CUDA 11.1.props")
#set_property(TARGET 5-add3memcpy APPEND PROPERTY VS_PROJECT_IMPORT "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\extras\\visual_studio_integration\\MSBuildExtensions\\CUDA 11.1.targets")
## target_compile_options(MyApp PIRVATE /bigobj /utf-8)
## target_link_options(MyApp PRIVATE /LARGEADDRESSAWARE)

#set_target_properties(5-add3memcpy PROPERTIES CUDA_ARCHITECTURES "35;52;75;80")


###############################

add_executable(5-arithmetic1cpu
    arithmetic1cpu.cu
)
target_include_directories(
    5-arithmetic1cpu PRIVATE
    ${CMAKE_SOURCE_DIR}
)
target_link_libraries(
    5-arithmetic1cpu PUBLIC
    -lcudart
    -lcudadevrt)

###############################

add_executable(5-arithmetic2gpu
    arithmetic2gpu.cu
)
target_include_directories(
    5-arithmetic2gpu PRIVATE
    ${CMAKE_SOURCE_DIR}
)
target_link_libraries(
    5-arithmetic2gpu PUBLIC
    -lcudart
    -lcudadevrt)