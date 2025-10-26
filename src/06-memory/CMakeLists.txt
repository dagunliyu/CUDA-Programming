
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

add_executable(6-query
    query.cu
)
target_include_directories(
    6-query PRIVATE
    ${CMAKE_SOURCE_DIR}
)
target_link_libraries(
    6-query PUBLIC
    -lcudart
    -lcudadevrt)

###############################

add_executable(6-static
    static.cu
)
target_include_directories(
    6-static PRIVATE
    ${CMAKE_SOURCE_DIR}
)
target_link_libraries(
    6-static PUBLIC
    -lcudart
    -lcudadevrt)

###############################