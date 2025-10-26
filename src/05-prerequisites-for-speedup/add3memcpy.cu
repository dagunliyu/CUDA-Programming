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

    const int block_size = 128;
    const int grid_size = (N + block_size - 1) / block_size;

    float t_sum = 0;
    float t2_sum = 0;
    for (int repeat = 0; repeat <= NUM_REPEATS; ++repeat)
    {
        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        CHECK(cudaEventRecord(start));
        cudaEventQuery(start);

        CHECK(cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_y, h_y, M, cudaMemcpyHostToDevice));
        add<<<grid_size, block_size>>>(d_x, d_y, d_z, N);
        CHECK(cudaMemcpy(h_z, d_z, M, cudaMemcpyDeviceToHost));

        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        float elapsed_time;
        CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
        printf("Time = %g ms.\n", elapsed_time);

        if (repeat > 0)
        {
            t_sum += elapsed_time;
            t2_sum += elapsed_time * elapsed_time;
        }

        CHECK(cudaEventDestroy(start));
        CHECK(cudaEventDestroy(stop));
    }

    const float t_ave = t_sum / NUM_REPEATS;
    const float t_err = sqrt(t2_sum / NUM_REPEATS - t_ave * t_ave);
    printf("Time = %g +- %g ms.\n", t_ave, t_err);

    //使用单精度时，
    //    数据复制和核函数调用共耗时 180 毫秒；使用双精度时，它们共耗时 360 毫秒。
    //    从上述测试得到的数据可以看到一个令人惊讶的结果：核函数的运行时间不到数据复
    //    制时间的 2 % 。如果将 CPU 与 GPU 之间的数据传输时间也计入，CUDA 程序相对于 C++ 程
    //    序得到的不是性能提升，而是性能降低。

    check(h_z, N);

    free(h_x);
    free(h_y);
    free(h_z);
    CHECK(cudaFree(d_x));
    CHECK(cudaFree(d_y));
    CHECK(cudaFree(d_z));
    return 0;
}

//如果一个程序的计算任务仅仅是将来自主机端的两个数组相加，并且要将结果传回主机端，使用 GPU 就不是一个明智的选择。那么，
//什么样的计算任务能够用 GPU 获得加速呢？

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

//$ nvprof . / a.out
//如果用上述命令时遇到了类似如下的错误提示：
//Unable to profile application.Unified Memory profiling failed
//则可以尝试将运行命令换为
//$ nvprof --unified - memory - profiling off . / a.out
//对程序 add3memcpy.cu 来说，在 GeForce RTX 2070 中使用上述命令，得到的部分结果如下
//（单精度浮点数版本）：
//Time(%) Time Calls Avg Min Max Name
//47.00 % 134.38ms 2 67.191ms 62.854ms 71.527ms[CUDA memcpy HtoD]
//40.13 % 114.74ms 1 114.74ms 114.74ms 114.74ms[CUDA memcpy DtoH]
//12.86 % 36.778ms 11 3.3435ms 3.3424ms 3.3501ms add()
//为排版方便起见，我们将 add() 函数中的参数类型省去了，而在原始的输出中函数的参数类型是保留的。
// 第一列是此处列出的每类操作所用时间的百分比，
// 第二列是每类操作用的总时间，
// 第三列是每类操作被调用的次数，
// 第四列是每类操作单次调用所用时间的平均值，
// 第五列是每类操作单次调用所用时间的最小值，
// 第六列是每类操作单次调用所用时间的最大值，
// 第七列是每类操作的名称。
// 从这里的输出可以看出核函数的执行时间及数据传输所用时间，它们和用 CUDA 事件获得的结果是一致的。