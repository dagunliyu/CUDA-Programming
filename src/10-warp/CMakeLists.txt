
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

add_executable(10-reduce
    reduce.cu
)
target_include_directories(
    10-reduce PRIVATE
    ${CMAKE_SOURCE_DIR}
)
target_link_libraries(
    10-reduce PUBLIC
    -lcudart
    -lcudadevrt)

###############################

add_executable(10-reduce1parallelism
    reduce1parallelism.cu
)
target_include_directories(
    10-reduce1parallelism PRIVATE
    ${CMAKE_SOURCE_DIR}
)
target_link_libraries(
    10-reduce1parallelism PUBLIC
    -lcudart
    -lcudadevrt)

###############################

add_executable(10-reduce2static
    reduce2static.cu
)
target_include_directories(
    10-reduce2static PRIVATE
    ${CMAKE_SOURCE_DIR}
)
target_link_libraries(
    10-reduce2static PUBLIC
    -lcudart
    -lcudadevrt)

###############################

add_executable(10-warp
    reduce.cu
)
target_include_directories(
    10-warp PRIVATE
    ${CMAKE_SOURCE_DIR}
)
target_link_libraries(
    10-warp PUBLIC
    -lcudart
    -lcudadevrt)

###############################