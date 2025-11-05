
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

add_executable(14-cublas_gemm
    cublas_gemm.cu
)
target_include_directories(
    14-cublas_gemm PRIVATE
    ${CMAKE_SOURCE_DIR}
)
target_link_libraries(
    14-cublas_gemm PUBLIC
    -lcudart
    -lcudadevrt)

###############################

add_executable(14-curand_host1
    curand_host1.cu
)
target_include_directories(
    14-curand_host1 PRIVATE
    ${CMAKE_SOURCE_DIR}
)
target_link_libraries(
    14-curand_host1 PUBLIC
    -lcudart
    -lcudadevrt)

###############################

add_executable(14-curand_host2
    curand_host2.cu
)
target_include_directories(
    14-curand_host2 PRIVATE
    ${CMAKE_SOURCE_DIR}
)
target_link_libraries(
    14-curand_host2 PUBLIC
    -lcudart
    -lcudadevrt)

###############################

add_executable(14-cusolver
    cusolver.cu
)
target_include_directories(
    14-cusolver PRIVATE
    ${CMAKE_SOURCE_DIR}
)
target_link_libraries(
    14-cusolver PUBLIC
    -lcudart
    -lcudadevrt)

###############################

add_executable(14-thrust_scan_pointer
    thrust_scan_pointer.cu
)
target_include_directories(
    14-thrust_scan_pointer PRIVATE
    ${CMAKE_SOURCE_DIR}
)
target_link_libraries(
    14-thrust_scan_pointer PUBLIC
    -lcudart
    -lcudadevrt)

###############################

add_executable(14-thrust_scan_vector
    thrust_scan_vector.cu
)
target_include_directories(
    14-thrust_scan_vector PRIVATE
    ${CMAKE_SOURCE_DIR}
)
target_link_libraries(
    14-thrust_scan_vector PUBLIC
    -lcudart
    -lcudadevrt)

###############################