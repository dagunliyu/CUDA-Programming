
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

add_executable(12-add
    add.cu
)
target_include_directories(
    12-add PRIVATE
    ${CMAKE_SOURCE_DIR}
)
target_link_libraries(
    12-add PUBLIC
    -lcudart
    -lcudadevrt)

###############################

add_executable(12-oversubscription1
    oversubscription1.cu
)
target_include_directories(
    12-oversubscription1 PRIVATE
    ${CMAKE_SOURCE_DIR}
)
target_link_libraries(
    12-oversubscription1 PUBLIC
    -lcudart
    -lcudadevrt)

###############################

add_executable(12-oversubscription2
    oversubscription1.cu
)
target_include_directories(
    12-oversubscription2 PRIVATE
    ${CMAKE_SOURCE_DIR}
)
target_link_libraries(
    12-oversubscription2 PUBLIC
    -lcudart
    -lcudadevrt)

###############################

add_executable(12-oversubscription3
    oversubscription1.cu
)
target_include_directories(
    12-oversubscription3 PRIVATE
    ${CMAKE_SOURCE_DIR}
)
target_link_libraries(
    12-oversubscription3 PUBLIC
    -lcudart
    -lcudadevrt)

###############################

add_executable(12-prefetch
    oversubscription1.cu
)
target_include_directories(
    12-prefetch PRIVATE
    ${CMAKE_SOURCE_DIR}
)
target_link_libraries(
    12-prefetch PUBLIC
    -lcudart
    -lcudadevrt)

###############################