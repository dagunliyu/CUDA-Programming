#include "error.cuh"
#include <math.h>
#include <stdio.h>

#ifdef USE_DP
    typedef double real;
    const real EPSILON = 1.0e-15;
#else
    typedef float real;
    const real EPSILON = 1.0e-6f;
#endif

const int NUM_REPEATS = 10;
const real a = 1.23;
const real b = 2.34;
const real c = 3.57;
void __global__ add(const real *x, const real *y, real *z, const int N);
void check(const real *z, const int N);

int main(void)
{
    const int N = 100000000;
    const int M = sizeof(real) * N;
    real *h_x = (real*) malloc(M);
    real *h_y = (real*) malloc(M);
    real *h_z = (real*) malloc(M);

    for (int n = 0; n < N; ++n)
    {
        h_x[n] = a;
        h_y[n] = b;
    }

    real *d_x, *d_y, *d_z;
     CHECK(cudaMalloc((void **)&d_x, M));
     CHECK(cudaMalloc((void **)&d_y, M));
     CHECK(cudaMalloc((void **)&d_z, M));
     CHECK(cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice));
     CHECK(cudaMemcpy(d_y, h_y, M, cudaMemcpyHostToDevice));

    const int block_size = 128;
    const int grid_size = (N + block_size - 1) / block_size;

    float t_sum = 0;
    float t2_sum = 0;
    for (int repeat = 0; repeat <= NUM_REPEATS; ++repeat)
    {
        cudaEvent_t start, stop;            // CUDA 事件类型（cudaEvent_t）
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        CHECK(cudaEventRecord(start));      // 记录一个代表开始的cuda事件
        
        // WDDM驱动模式的GPU中, 一个CUDA流(CUDA Stream)中的操作（如这里的cudaEventQuery），不是直接提交给GPU执行
        // 而是先提交到一个软件队列，需要添加一条对该流的cudaEventQuery操作刷新队列，才能促使 前面的操作在GPU中执行
        cudaEventQuery(start);
        // 此处不能用 CHECK 宏函数: 因为它很有可能返回 cudaErrorNotReady，但又不代表程序出错了

        add<<<grid_size, block_size>>>(d_x, d_y, d_z, N);

        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        float elapsed_time;
        CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
        printf("Time = %g ms.\n", elapsed_time);

        //忽略第一次测得的时间，因为第一次计算时，机器（无论是 CPU 还是 GPU）都可能
        //处于预热状态，测得的时间往往偏大。我们根据后 10 次测试的时间计算一个平均值。
        if (repeat > 0)
        {
            t_sum += elapsed_time;
            t2_sum += elapsed_time * elapsed_time;
        }

        CHECK(cudaEventDestroy(start));
        CHECK(cudaEventDestroy(stop));
    }

    //使用单精度浮点数时核函数 add 所用时间约为 3.3 ms，
    //使用双精度浮点数时核函数 add 所用时间约为 6.8 ms。这两个时间的比值也约为 2。

    //从表 5.1 中可以看出，该比值与单、双精度浮点数运算峰值的比值没有关系。
    //这是因为，对于数组相加的问题，其执行速度是由显存带宽决定的，而不是由浮点数运算峰值决定的。
    const float t_ave = t_sum / NUM_REPEATS;
    const float t_err = sqrt(t2_sum / NUM_REPEATS - t_ave * t_ave);
    printf("Time = %g +- %g ms.\n", t_ave, t_err);

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

// 计算数组相加问题在 GPU 中达到的有效显存带宽（effective memory bandwidth），并与表 5.1 中的理论显存带宽（theoretical memory bandwidth）进行比较。
// 有效显存带宽定义为 GPU 在单位时间内访问设备内存的字节数。


//有效显存带宽略小于理论显存带宽，进一步说明该问题是访存主导的，即该问题中
//的浮点数运算所占比例可以忽略不计
void __global__ add(const real *x, const real *y, real *z, const int N)
{
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n < N)
    {
        z[n] = x[n] + y[n];
    }
}

void check(const real *z, const int N)
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

