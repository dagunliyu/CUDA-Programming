
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

add_executable(8-bank
    bank.cu
)
target_include_directories(
    8-bank PRIVATE
    ${CMAKE_SOURCE_DIR}
)
target_link_libraries(
    8-bank PUBLIC
    -lcudart
    -lcudadevrt)

###############################

add_executable(8-reduce1cpu
    reduce1cpu.cu
)
target_include_directories(
    8-reduce1cpu PRIVATE
    ${CMAKE_SOURCE_DIR}
    )
target_link_libraries(
    8-reduce1cpu PUBLIC
    -lcudart
    -lcudadevrt)

###############################
add_executable(8-reduce2gpu
    reduce2gpu.cu
)
target_include_directories(
    8-reduce1cpu PRIVATE
    ${CMAKE_SOURCE_DIR}
    )
target_link_libraries(
    8-reduce2gpu PUBLIC
    -lcudart
    -lcudadevrt)

###############################