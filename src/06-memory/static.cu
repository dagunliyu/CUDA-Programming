#include "error.cuh"
#include <stdio.h>

// 静态内存全局变量
// __device__ 用于声明设备端变量而非主机端的变量，这些变量存储在GPU的全局内存中
// 占用内存数量在编译时确定，必须在所有主机和设备外部定义，非动态分配
__device__ int d_x = 1; // 单个变量
__device__ int d_y[2];  // 固定长度的数组

void __global__ my_kernel(void)
{
    d_y[0] += d_x;
    d_y[1] += d_x;
    printf("d_x = %d, d_y[0] = %d, d_y[1] = %d.\n", d_x, d_y[0], d_y[1]);
}

int main(void)
{
    int h_y[2] = {10, 20};
	// 将主机端数据复制到设备端静态内存变量中
    CHECK(cudaMemcpyToSymbol(d_y, h_y, sizeof(int) * 2));
    
    my_kernel<<<1, 1>>>();
    CHECK(cudaDeviceSynchronize());
    
	// 将设备端静态内存变量的数据复制回主机端
    CHECK(cudaMemcpyFromSymbol(h_y, d_y, sizeof(int) * 2));
    printf("h_y[0] = %d, h_y[1] = %d.\n", h_y[0], h_y[1]);
    
    return 0;
}

