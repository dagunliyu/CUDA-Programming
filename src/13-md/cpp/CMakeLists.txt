
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

add_executable(13-cpp
    main.cu
)
target_include_directories(
    13-cpp PRIVATE
    ${CMAKE_SOURCE_DIR}
)
target_link_libraries(
    13-cpp PUBLIC
    -lcudart
    -lcudadevrt)

###############################