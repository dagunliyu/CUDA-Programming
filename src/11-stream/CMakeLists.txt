
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

add_executable(11-host-kernel
    host-kernel.cu
)
target_include_directories(
    11-host-kernel PRIVATE
    ${CMAKE_SOURCE_DIR}
)
target_link_libraries(
    11-host-kernel PUBLIC
    -lcudart
    -lcudadevrt)

###############################

add_executable(11-kernel-kernel
    kernel-kernel.cu
)
target_include_directories(
    11-kernel-kernel PRIVATE
    ${CMAKE_SOURCE_DIR}
)
target_link_libraries(
    11-kernel-kernel PUBLIC
    -lcudart
    -lcudadevrt)

###############################

add_executable(11-kernel-transfer
    kernel-transfer.cu
)
target_include_directories(
    11-kernel-transfer PRIVATE
    ${CMAKE_SOURCE_DIR}
)
target_link_libraries(
    11-kernel-transfer PUBLIC
    -lcudart
    -lcudadevrt)
 
###############################