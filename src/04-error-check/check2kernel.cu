#include "error.cuh"
#include <math.h>
#include <stdio.h>

const double EPSILON = 1.0e-15;
const double a = 1.23;
const double b = 2.34;
const double c = 3.57;
void __global__ add(const double *x, const double *y, double *z, const int N);
void check(const double *z, const int N);

int main(void)
{
    const int N = 100000000;
    const int M = sizeof(double) * N;
    double *h_x = (double*) malloc(M);
    double *h_y = (double*) malloc(M);
    double *h_z = (double*) malloc(M);

    for (int n = 0; n < N; ++n)
    {
        h_x[n] = a;
        h_y[n] = b;
    }

    //有一个例外是 cudaEventQuery 函数，因为它很有可能返回 cudaErrorNotReady，但又不代表程序出错了

    double *d_x, *d_y, *d_z;
    CHECK(cudaMalloc((void **)&d_x, M));
    CHECK(cudaMalloc((void **)&d_y, M));
    CHECK(cudaMalloc((void **)&d_z, M));
    CHECK(cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_y, h_y, M, cudaMemcpyHostToDevice));

    // 也可以换成<<<1, 1>>>,只用1个线程运行，此时就可以把kernel中的代码换成主机中的代码便于进行测试

    // 10000001/128 = 781250 余数为1, 若取gridsize=781250，则线程数一共就10^8，还少1个
    // 取gridsize=781252，这样线程数是10^8+128，能覆盖100000001，只是多了127个线程
    // 可通过if来规避不需要的线程

    // 当N是blocksize(1个block中的thread数量)的整数倍
    // 若不是整数倍,就可能引发错误

    const int block_size = 1280;
    const int grid_size = (N + block_size - 1) / block_size;
    add<<<grid_size, block_size>>>(d_x, d_y, d_z, N);
    //线程块大小的最大值是 1024（这对从开普勒到图灵的所有架构都成立）。假如我们不小心将核函数执行配置中
    //的线程块大小写成了 1280，该核函数将不能被成功地调用。


    CHECK(cudaGetLastError());  // 捕捉下句之前的最后一个错误
    CHECK(cudaDeviceSynchronize());
    //同步主机与设备。之所以要同步主机与设备，是因为核函数的调用是异步的，即主机发出调用核函数的命令后会立即执行后面的语句，不会等待核函数执行完毕

    //上述同步函数是比较耗时的，如果在程序的较内层循环调用的话，很可能会严重地降低程序的性能。
    //所以，一般不在程序的较内层循环调用上述同步函数。
    //只要在核函数的调用之后还有对其他任何能返回错误值的 API 函数进行同步调用，都能够触发主机与设备的同步并捕捉到核函数调用中可能发生的错误


    CHECK(cudaMemcpy(h_z, d_z, M, cudaMemcpyDeviceToHost));
    check(h_z, N);

    free(h_x);
    free(h_y);
    free(h_z);
    CHECK(cudaFree(d_x));
    CHECK(cudaFree(d_y));
    CHECK(cudaFree(d_z));
    return 0;
}

void __global__ add(const double *x, const double *y, double *z, const int N)
{
    // blockDim.x：对于分配的一维数组来说,表示每个block的size
    // blockIdx.x：当前线程所在的block的idx
    // threadIdx.x: 当前线程在当前block中的idx
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n < N)
    {
        // 10000001/128 = 781250 余数为1, 若取gridsize=781250，则线程数一共就10^8，还少1个
        // 取gridsize=781252，这样线程数是10^8+128，能覆盖100000001，只是多了127个线程
        // 可通过if来规避不需要的线程，这样只有100000001个线程会生效，其他的线程不会执行到这个if里
        z[n] = x[n] + y[n];
    }

    // 第0号thread-block，包含第0~blockDim.x-1个线程(数组元素)
    // 第1号线程块：包含第blockDim.x ~ 2*blockDim.x-1个数组元素(线程)
}

void check(const double *z, const int N)
{
    bool has_error = false;
    for (int n = 0; n < N; ++n)
    {
        if (fabs(z[n] - c) > EPSILON)
        {
            has_error = true;
        }
    }
    printf("%s\n", has_error ? "Has errors" : "No errors");
}

