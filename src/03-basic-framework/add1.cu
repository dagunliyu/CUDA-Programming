#include <math.h>
#include <stdio.h>

const double EPSILON = 1.0e-15;
const double a = 1.23;
const double b = 2.34;
const double c = 3.57;
void __global__ add(const double *x, const double *y, double *z);
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

    // 内存示意图：
    // +---------+          +-------------------+
    // |         | 指向     |                   |
    // | &d_x    | -------> | d_x (指针变量)    |
    // | (double**)|        | (存储设备内存地址) |
    // +---------+          +-------------------+
    //                          |
    //                          | 当 cudaMalloc 执行后
    //                          v
    //                     [GPU 内存区块]

    // d_x的类型是double*, 取地址为&d_x,对应的类型为double**
    // d_x本身表示数据的地址, *d_x是取这个地址里的数据的值; 
    // &d_x获取的是存储这个地址的地址
    // &(*d_x) = d_x
    // 使用void**可以接受任意类型的指针, 提供了通用解决方案
    // 因为内存地址本身就是一个指针，所以待分配设备内存的指针就是指针的指针，即双重指针
    // cudaMalloc要改指针d_x本身的值(将一个指针赋值给d_x), 而不是改变d_x所指内存缓冲区中的变量值
    double *d_x, *d_y, *d_z;
    cudaMalloc((void **)&d_x, M);  // cudaMalloc会修改d_x的地址, 确保能正确的把device内存地址赋值给d_x
    cudaMalloc((void **)&d_y, M);  // 用cudaMalloc在GPU上分配内存, 并将分配的地址存回d_x
    cudaMalloc((void **)&d_z, M);
    cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice); // 此时的d_x已经指向设备内存
    cudaMemcpy(d_y, h_y, M, cudaMemcpyHostToDevice);

    // N为10^8, 则grid_size为781250
    // 若用CUDA8.0, 若在nvcc编译时没有指定计算能力, 那么会使用默认的2.0的计算能力, 2.0计算能力的网格大小在x方向上的上限是65535. 会影响计算
    const int block_size = 128;
    const int grid_size = N / block_size;
    add<<<grid_size, block_size>>>(d_x, d_y, d_z);


    cudaMemcpy(h_z, d_z, M, cudaMemcpyDeviceToHost);
    check(h_z, N);

    free(h_x);
    free(h_y);
    free(h_z);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
    return 0;


    // 流程
    // 分配host和device的内存
    // 将需要的数据从host复制到device
    // 利用kernel进行计算
    // 将计算结果从device复制到host
    // 释放host和device的内存
}

void __global__ add(const double *x, const double *y, double *z)
{
    // blockDim.x：对于分配的一维数组来说,表示每个block的size
    // blockIdx.x：当前线程所在的block的idx
    // threadIdx.x: 当前线程在当前block中的idx
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    z[n] = x[n] + y[n];

    // 计算能力2.0开始, cuda允许kernel中用malloc和free动态分配和释放内存. 但是容易导致较差的程序性能不建议使用。
    // 如果有这样的需求，可能需要思考如何重构算法
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

