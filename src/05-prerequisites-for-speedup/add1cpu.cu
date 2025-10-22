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
void add(const real *x, const real *y, real *z, const int N);
void check(const real *z, const int N);

int main(void)
{
    const int N = 100000000;
    const int M = sizeof(real) * N;
    real *x = (real*) malloc(M);
    real *y = (real*) malloc(M);
    real *z = (real*) malloc(M);

    for (int n = 0; n < N; ++n)
    {
        x[n] = a;
        y[n] = b;
    }

    float t_sum = 0;
    float t2_sum = 0;
    //重复了 11 次
    for (int repeat = 0; repeat <= NUM_REPEATS; ++repeat)
    {
        cudaEvent_t start, stop;        // CUDA 事件类型（cudaEvent_t）
        CHECK(cudaEventCreate(&start));  
        CHECK(cudaEventCreate(&stop));  
        CHECK(cudaEventRecord(start));  // 记录一个代表开始的cuda事件
        
        // WDDM驱动模式的GPU中, 一个CUDA流(CUDA Stream)中的操作（如这里的cudaEventQuery），不是直接提交给GPU执行
        // 而是先提交到一个软件队列，需要添加一条对该流的cudaEventQuery操作刷新队列，才能促使 前面的操作在GPU中执行
        cudaEventQuery(start);          // 此处不能用 CHECK 宏函数: 因为它很有可能返回 cudaErrorNotReady，但又不代表程序出错了

        add(x, y, z, N);

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

    //我们根据后 10 次测试的时间计算一个平均值。
    const float t_ave = t_sum / NUM_REPEATS;
    const float t_err = sqrt(t2_sum / NUM_REPEATS - t_ave * t_ave);
    printf("Time = %g +- %g ms.\n", t_ave, t_err);
    
    //add 函数耗时约 120 ms。我们看到，双精度版本的 add 函数所用时间大概是单精度版本的 add 函数所用时间的 2 倍，
    //这对于这种访存主导的函数来说是合理的。


    check(z, N);

    free(x);
    free(y);
    free(z);
    return 0;
}

void add(const real *x, const real *y, real *z, const int N)
{
    for (int n = 0; n < N; ++n)
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


