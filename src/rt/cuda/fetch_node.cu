#include "cutil_inline_runtime.h"

#include "base/Defs.hh"
#include "base/Math.hh"
#include "Util.hh"
#include "gpu/CudaKernel.hh"
#include "kernels/CudaTracerKernels.hh"

using namespace FW;

#define node_nums 301376

__global__ void fetch_n0xy_x(float* n0xy_x, float4* node_a_addr)
{
    for (int i = 0 ; i < node_nums; i++){
        float4 n0xy = FETCH_GLOBAL(node_a_addr, i * 4 , float4);
        n0xy_x[i]  = n0xy.x;
        // printf("n0xy_x[%d]  = %X\n", i + 1, __float_as_int(n0xy_x[i]));
    }
}

extern "C"  float* fetch_n0xyx(float4* node_a_addr)
{
    float* n0xy_x;
    cudaMalloc((void**)&n0xy_x, sizeof(float)* node_nums);//(void**)&  其中(void **)是一个强制类型转换的。也就是把一个东西变为指向指针的指针    
    fetch_n0xy_x<<<1,1>>>(n0xy_x,node_a_addr);
    float* n0xy_x_out;
    n0xy_x_out = (float*)malloc(sizeof(float)*node_nums);
    cudaMemcpy(n0xy_x_out, n0xy_x, sizeof(float)*node_nums, cudaMemcpyDeviceToHost);
    cudaFree(n0xy_x);
    return n0xy_x_out;
}

__global__ void fetch_n0xy_y(float* n0xy_y, float4* node_a_addr)
{
    for (int i = 0 ; i < node_nums; i++){
        float4 n0xy = FETCH_GLOBAL(node_a_addr, i * 4 , float4);
        n0xy_y[i]  = n0xy.y;
        // printf("n0xy_x[%d]  = %X\n", i + 1, __float_as_int(n0xy_x[i]));
    }
}

extern "C"  float* fetch_n0xyy(float4* node_a_addr)
{
    float* n0xy_y;
    cudaMalloc((void**)&n0xy_y, sizeof(float)* node_nums);//(void**)&  其中(void **)是一个强制类型转换的。也就是把一个东西变为指向指针的指针    
    fetch_n0xy_y<<<1,1>>>(n0xy_y,node_a_addr);
    float* n0xy_y_out;
    n0xy_y_out = (float*)malloc(sizeof(float)*node_nums);
    cudaMemcpy(n0xy_y_out, n0xy_y, sizeof(float)*node_nums, cudaMemcpyDeviceToHost);
    cudaFree(n0xy_y);
    return n0xy_y_out;
}

__global__ void fetch_n0xy_z(float* n0xy_z, float4* node_a_addr)
{
    for (int i = 0 ; i < node_nums; i++){
        float4 n0xy = FETCH_GLOBAL(node_a_addr, i * 4 , float4);
        n0xy_z[i]  = n0xy.z;
        // printf("n0xy_x[%d]  = %X\n", i + 1, __float_as_int(n0xy_x[i]));
    }
}

extern "C"  float* fetch_n0xyz(float4* node_a_addr)
{
    float* n0xy_z;
    cudaMalloc((void**)&n0xy_z, sizeof(float)* node_nums);//(void**)&  其中(void **)是一个强制类型转换的。也就是把一个东西变为指向指针的指针    
    fetch_n0xy_z<<<1,1>>>(n0xy_z,node_a_addr);
    float* n0xy_z_out;
    n0xy_z_out = (float*)malloc(sizeof(float)*node_nums);
    cudaMemcpy(n0xy_z_out, n0xy_z, sizeof(float)*node_nums, cudaMemcpyDeviceToHost);
    cudaFree(n0xy_z);
    return n0xy_z_out;
}

__global__ void fetch_n0xy_w(float* n0xy_w, float4* node_a_addr)
{
    for (int i = 0 ; i < node_nums; i++){
        float4 n0xy = FETCH_GLOBAL(node_a_addr, i * 4 , float4);
        n0xy_w[i]  = n0xy.w;
        // printf("n0xy_x[%d]  = %X\n", i + 1, __float_as_int(n0xy_x[i]));
    }
}

extern "C"  float* fetch_n0xyw(float4* node_a_addr)
{
    float* n0xy_w;
    cudaMalloc((void**)&n0xy_w, sizeof(float)* node_nums);//(void**)&  其中(void **)是一个强制类型转换的。也就是把一个东西变为指向指针的指针    
    fetch_n0xy_w<<<1,1>>>(n0xy_w,node_a_addr);
    float* n0xy_w_out;
    n0xy_w_out = (float*)malloc(sizeof(float)*node_nums);
    cudaMemcpy(n0xy_w_out, n0xy_w, sizeof(float)*node_nums, cudaMemcpyDeviceToHost);
    cudaFree(n0xy_w);
    return n0xy_w_out;
}

//-----------------------------------------------------------n1xy-------------------------------------------------------

__global__ void fetch_n1xy_x(float* n1xy_x, float4* node_a_addr)
{
    for (int i = 0 ; i < node_nums; i++){
        float4 n1xy = FETCH_GLOBAL(node_a_addr, i * 4 + 1 , float4);
        n1xy_x[i]  = n1xy.x;
        // printf("n0xy_x[%d]  = %X\n", i + 1, __float_as_int(n0xy_x[i]));
    }
}

extern "C"  float* fetch_n1xyx(float4* node_a_addr)
{
    float* n1xy_x;
    cudaMalloc((void**)&n1xy_x, sizeof(float)* node_nums);//(void**)&  其中(void **)是一个强制类型转换的。也就是把一个东西变为指向指针的指针    
    fetch_n1xy_x<<<1,1>>>(n1xy_x,node_a_addr);
    float* n1xy_x_out;
    n1xy_x_out = (float*)malloc(sizeof(float)*node_nums);
    cudaMemcpy(n1xy_x_out, n1xy_x, sizeof(float)*node_nums, cudaMemcpyDeviceToHost);
    cudaFree(n1xy_x);
    return n1xy_x_out;
}

__global__ void fetch_n1xy_y(float* n1xy_y, float4* node_a_addr)
{
    for (int i = 0 ; i < node_nums; i++){
        float4 n1xy = FETCH_GLOBAL(node_a_addr, i * 4 + 1 , float4);
        n1xy_y[i]  = n1xy.y;
        // printf("n0xy_x[%d]  = %X\n", i + 1, __float_as_int(n0xy_x[i]));
    }
}

extern "C"  float* fetch_n1xyy(float4* node_a_addr)
{
    float* n1xy_y;
    cudaMalloc((void**)&n1xy_y, sizeof(float)* node_nums);//(void**)&  其中(void **)是一个强制类型转换的。也就是把一个东西变为指向指针的指针    
    fetch_n1xy_y<<<1,1>>>(n1xy_y,node_a_addr);
    float* n1xy_y_out;
    n1xy_y_out = (float*)malloc(sizeof(float)*node_nums);
    cudaMemcpy(n1xy_y_out, n1xy_y, sizeof(float)*node_nums, cudaMemcpyDeviceToHost);
    cudaFree(n1xy_y);
    return n1xy_y_out;
}

__global__ void fetch_n1xy_z(float* n1xy_z, float4* node_a_addr)
{
    for (int i = 0 ; i < node_nums; i++){
        float4 n1xy = FETCH_GLOBAL(node_a_addr, i * 4 + 1 , float4);
        n1xy_z[i]  = n1xy.z;
        // printf("n0xy_x[%d]  = %X\n", i + 1, __float_as_int(n0xy_x[i]));
    }
}

extern "C"  float* fetch_n1xyz(float4* node_a_addr)
{
    float* n1xy_z;
    cudaMalloc((void**)&n1xy_z, sizeof(float)* node_nums);//(void**)&  其中(void **)是一个强制类型转换的。也就是把一个东西变为指向指针的指针    
    fetch_n1xy_z<<<1,1>>>(n1xy_z,node_a_addr);
    float* n1xy_z_out;
    n1xy_z_out = (float*)malloc(sizeof(float)*node_nums);
    cudaMemcpy(n1xy_z_out, n1xy_z, sizeof(float)*node_nums, cudaMemcpyDeviceToHost);
    cudaFree(n1xy_z);
    return n1xy_z_out;
}

__global__ void fetch_n1xy_w(float* n1xy_w, float4* node_a_addr)
{
    for (int i = 0 ; i < node_nums; i++){
        float4 n1xy = FETCH_GLOBAL(node_a_addr, i * 4 + 1 , float4);
        n1xy_w[i]  = n1xy.w;
        // printf("n0xy_x[%d]  = %X\n", i + 1, __float_as_int(n0xy_x[i]));
    }
}

extern "C"  float* fetch_n1xyw(float4* node_a_addr)
{
    float* n1xy_w;
    cudaMalloc((void**)&n1xy_w, sizeof(float)* node_nums);//(void**)&  其中(void **)是一个强制类型转换的。也就是把一个东西变为指向指针的指针    
    fetch_n1xy_w<<<1,1>>>(n1xy_w,node_a_addr);
    float* n1xy_w_out;
    n1xy_w_out = (float*)malloc(sizeof(float)*node_nums);
    cudaMemcpy(n1xy_w_out, n1xy_w, sizeof(float)*node_nums, cudaMemcpyDeviceToHost);
    cudaFree(n1xy_w);
    return n1xy_w_out;
}

//-----------------------------------------------------------n2xy-------------------------------------------------------

__global__ void fetch_n2xy_x(float* n2xy_x, float4* node_a_addr)
{
    for (int i = 0 ; i < node_nums; i++){
        float4 n2xy = FETCH_GLOBAL(node_a_addr, i * 4 + 2 , float4);
        n2xy_x[i]  = n2xy.x;
        // printf("n0xy_x[%d]  = %X\n", i + 1, __float_as_int(n0xy_x[i]));
    }
}

extern "C"  float* fetch_n2xyx(float4* node_a_addr)
{
    float* n2xy_x;
    cudaMalloc((void**)&n2xy_x, sizeof(float)* node_nums);//(void**)&  其中(void **)是一个强制类型转换的。也就是把一个东西变为指向指针的指针    
    fetch_n2xy_x<<<1,1>>>(n2xy_x,node_a_addr);
    float* n2xy_x_out;
    n2xy_x_out = (float*)malloc(sizeof(float)*node_nums);
    cudaMemcpy(n2xy_x_out, n2xy_x, sizeof(float)*node_nums, cudaMemcpyDeviceToHost);
    cudaFree(n2xy_x);
    return n2xy_x_out;
}

__global__ void fetch_n2xy_y(float* n2xy_y, float4* node_a_addr)
{
    for (int i = 0 ; i < node_nums; i++){
        float4 n2xy = FETCH_GLOBAL(node_a_addr, i * 4 + 2 , float4);
        n2xy_y[i]  = n2xy.y;
        // printf("n0xy_x[%d]  = %X\n", i + 1, __float_as_int(n0xy_x[i]));
    }
}

extern "C"  float* fetch_n2xyy(float4* node_a_addr)
{
    float* n2xy_y;
    cudaMalloc((void**)&n2xy_y, sizeof(float)* node_nums);//(void**)&  其中(void **)是一个强制类型转换的。也就是把一个东西变为指向指针的指针    
    fetch_n2xy_y<<<1,1>>>(n2xy_y,node_a_addr);
    float* n2xy_y_out;
    n2xy_y_out = (float*)malloc(sizeof(float)*node_nums);
    cudaMemcpy(n2xy_y_out, n2xy_y, sizeof(float)*node_nums, cudaMemcpyDeviceToHost);
    cudaFree(n2xy_y);
    return n2xy_y_out;
}

__global__ void fetch_n2xy_z(float* n2xy_z, float4* node_a_addr)
{
    for (int i = 0 ; i < node_nums; i++){
        float4 n2xy = FETCH_GLOBAL(node_a_addr, i * 4 + 2 , float4);
        n2xy_z[i]  = n2xy.z;
        // printf("n0xy_x[%d]  = %X\n", i + 1, __float_as_int(n0xy_x[i]));
    }
}

extern "C"  float* fetch_n2xyz(float4* node_a_addr)
{
    float* n2xy_z;
    cudaMalloc((void**)&n2xy_z, sizeof(float)* node_nums);//(void**)&  其中(void **)是一个强制类型转换的。也就是把一个东西变为指向指针的指针    
    fetch_n2xy_z<<<1,1>>>(n2xy_z,node_a_addr);
    float* n2xy_z_out;
    n2xy_z_out = (float*)malloc(sizeof(float)*node_nums);
    cudaMemcpy(n2xy_z_out, n2xy_z, sizeof(float)*node_nums, cudaMemcpyDeviceToHost);
    cudaFree(n2xy_z);
    return n2xy_z_out;
}

__global__ void fetch_n2xy_w(float* n2xy_w, float4* node_a_addr)
{
    for (int i = 0 ; i < node_nums; i++){
        float4 n2xy = FETCH_GLOBAL(node_a_addr, i * 4 + 2 , float4);
        n2xy_w[i]  = n2xy.w;
        // printf("n0xy_x[%d]  = %X\n", i + 1, __float_as_int(n0xy_x[i]));
    }
}

extern "C"  float* fetch_n2xyw(float4* node_a_addr)
{
    float* n2xy_w;
    cudaMalloc((void**)&n2xy_w, sizeof(float)* node_nums);//(void**)&  其中(void **)是一个强制类型转换的。也就是把一个东西变为指向指针的指针    
    fetch_n2xy_w<<<1,1>>>(n2xy_w,node_a_addr);
    float* n2xy_w_out;
    n2xy_w_out = (float*)malloc(sizeof(float)*node_nums);
    cudaMemcpy(n2xy_w_out, n2xy_w, sizeof(float)*node_nums, cudaMemcpyDeviceToHost);
    cudaFree(n2xy_w);
    return n2xy_w_out;
}


//-----------------------------------------------------------childIndex-------------------------------------------------------

__global__ void fetch_child_x(int* child_x, float4* node_a_addr)
{
    for (int i = 0 ; i < node_nums; i++){
        float4 tmp = FETCH_GLOBAL(node_a_addr, i * 4 + 3 , float4);
        int2 cnodes= *(int2*)&tmp;
        child_x[i]  = cnodes.x;
        // printf("n0xy_x[%d]  = %X\n", i + 1, __float_as_int(n0xy_x[i]));
    }
}

extern "C"  int* fetch_child_x(float4* node_a_addr)
{
    int* child_x;
    cudaMalloc((void**)&child_x, sizeof(int)* node_nums);//(void**)&  其中(void **)是一个强制类型转换的。也就是把一个东西变为指向指针的指针    
    fetch_child_x<<<1,1>>>(child_x,node_a_addr);
    int* child_x_out;
    child_x_out = (int*)malloc(sizeof(int)*node_nums);
    cudaMemcpy(child_x_out, child_x, sizeof(int)*node_nums, cudaMemcpyDeviceToHost);
    cudaFree(child_x);
    return child_x_out;
}

__global__ void fetch_child_y(int* child_y, float4* node_a_addr)
{
    for (int i = 0 ; i < node_nums; i++){
        float4 tmp = FETCH_GLOBAL(node_a_addr, i * 4 + 3 , float4);
        int2 cnodes= *(int2*)&tmp;
        child_y[i]  = cnodes.y;
        // printf("n0xy_x[%d]  = %X\n", i + 1, __float_as_int(n0xy_x[i]));
    }
}

extern "C"  int* fetch_child_y(float4* node_a_addr)
{
    int* child_y;
    cudaMalloc((void**)&child_y, sizeof(int)* node_nums);//(void**)&  其中(void **)是一个强制类型转换的。也就是把一个东西变为指向指针的指针    
    fetch_child_y<<<1,1>>>(child_y,node_a_addr);
    int* child_y_out;
    child_y_out = (int*)malloc(sizeof(int)*node_nums);
    cudaMemcpy(child_y_out, child_y, sizeof(int)*node_nums, cudaMemcpyDeviceToHost);
    cudaFree(child_y);
    return child_y_out;
}
