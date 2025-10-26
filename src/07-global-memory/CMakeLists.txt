
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

add_executable(7-matrix
    matrix.cu
)
target_include_directories(
    7-matrix PRIVATE
    ${CMAKE_SOURCE_DIR}
)
target_link_libraries(
    7-matrix PUBLIC
    -lcudart
    -lcudadevrt)

###############################