
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

add_executable(9-neighbor1cpu
    neighbor1cpu.cu
)
target_include_directories(
    9-neighbor1cpu PRIVATE
    ${CMAKE_SOURCE_DIR}
)
target_link_libraries(
    9-neighbor1cpu PUBLIC
    -lcudart
    -lcudadevrt)

###############################

add_executable(9-neighbor2gpu
    neighbor2gpu.cu
)
target_include_directories(
    9-neighbor2gpu PRIVATE
    ${CMAKE_SOURCE_DIR}
)
target_link_libraries(
    9-neighbor2gpu PUBLIC
    -lcudart
    -lcudadevrt)

###############################

add_executable(9-reduce
    reduce.cu
)
target_include_directories(
    9-reduce PRIVATE
    ${CMAKE_SOURCE_DIR}
)
target_link_libraries(
    9-reduce PUBLIC
    -lcudart
    -lcudadevrt)

###############################