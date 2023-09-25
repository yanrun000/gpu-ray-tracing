#include "cutil_inline_runtime.h"

#include "base/Defs.hh"
#include "base/Math.hh"
#include "Util.hh"
#include "gpu/CudaKernel.hh"
#include "kernels/CudaTracerKernels.hh"

using namespace FW;

#define tri_nums 250000

__global__ void fetch_v00_x(float* v00_x, float4* tri_a_addr)
{
    for (int i = 0 ; i < tri_nums; i++){
        float4 v00 = FETCH_GLOBAL(tri_a_addr, i * 3 , float4);
        v00_x[i]  = v00.x;
    }
}

extern "C"  float* fetch_v00_x(float4* tri_a_addr)
{
    float* v00_x;
    cudaMalloc((void**)&v00_x, sizeof(float)* tri_nums);//(void**)&  其中(void **)是一个强制类型转换的。也就是把一个东西变为指向指针的指针    
    fetch_v00_x<<<1,1>>>(v00_x,tri_a_addr);
    float* v00_x_out;
    v00_x_out = (float*)malloc(sizeof(float)*tri_nums);
    cudaMemcpy(v00_x_out, v00_x, sizeof(float)*tri_nums, cudaMemcpyDeviceToHost);
    cudaFree(v00_x);
    return v00_x_out;
}

__global__ void fetch_v11_x(float* v11_x, float4* tri_a_addr)
{
    for (int i = 0 ; i < tri_nums; i++){
        float4 v11 = FETCH_GLOBAL(tri_a_addr, i * 3 + 1 , float4);
        v11_x[i]  = v11.x;
    }
}

extern "C"  float* fetch_v11_x(float4* tri_a_addr)
{
    float* v11_x;
    cudaMalloc((void**)&v11_x, sizeof(float)* tri_nums);//(void**)&  其中(void **)是一个强制类型转换的。也就是把一个东西变为指向指针的指针    
    fetch_v11_x<<<1,1>>>(v11_x,tri_a_addr);
    float* v11_x_out;
    v11_x_out = (float*)malloc(sizeof(float)*tri_nums);
    cudaMemcpy(v11_x_out, v11_x, sizeof(float)*tri_nums, cudaMemcpyDeviceToHost);
    cudaFree(v11_x);
    return v11_x_out;
}

__global__ void fetch_v22_x(float* v22_x, float4* tri_a_addr)
{
    for (int i = 0 ; i < tri_nums; i++){
        float4 v22 = FETCH_GLOBAL(tri_a_addr, i * 3 + 2 , float4);
        v22_x[i]  = v22.x;
    }
}

extern "C"  float* fetch_v22_x(float4* tri_a_addr)
{
    float* v22_x;
    cudaMalloc((void**)&v22_x, sizeof(float)* tri_nums);//(void**)&  其中(void **)是一个强制类型转换的。也就是把一个东西变为指向指针的指针    
    fetch_v22_x<<<1,1>>>(v22_x,tri_a_addr);
    float* v22_x_out;
    v22_x_out = (float*)malloc(sizeof(float)*tri_nums);
    cudaMemcpy(v22_x_out, v22_x, sizeof(float)*tri_nums, cudaMemcpyDeviceToHost);
    cudaFree(v22_x);
    return v22_x_out;
}


//---------------------------------------------------------------triangle_y---------------------------------------------------------
__global__ void fetch_v00_y(float* v00_y, float4* tri_a_addr)
{
    for (int i = 0 ; i < tri_nums; i++){
        float4 v00 = FETCH_GLOBAL(tri_a_addr, i * 3 , float4);
        v00_y[i]  = v00.y;
    }
}

extern "C"  float* fetch_v00_y(float4* tri_a_addr)
{
    float* v00_y;
    cudaMalloc((void**)&v00_y, sizeof(float)* tri_nums);//(void**)&  其中(void **)是一个强制类型转换的。也就是把一个东西变为指向指针的指针    
    fetch_v00_y<<<1,1>>>(v00_y,tri_a_addr);
    float* v00_y_out;
    v00_y_out = (float*)malloc(sizeof(float)*tri_nums);
    cudaMemcpy(v00_y_out, v00_y, sizeof(float)*tri_nums, cudaMemcpyDeviceToHost);
    cudaFree(v00_y);
    return v00_y_out;
}

__global__ void fetch_v11_y(float* v11_y, float4* tri_a_addr)
{
    for (int i = 0 ; i < tri_nums; i++){
        float4 v11 = FETCH_GLOBAL(tri_a_addr, i * 3 + 1 , float4);
        v11_y[i]  = v11.y;
    }
}

extern "C"  float* fetch_v11_y(float4* tri_a_addr)
{
    float* v11_y;
    cudaMalloc((void**)&v11_y, sizeof(float)* tri_nums);//(void**)&  其中(void **)是一个强制类型转换的。也就是把一个东西变为指向指针的指针    
    fetch_v11_y<<<1,1>>>(v11_y,tri_a_addr);
    float* v11_y_out;
    v11_y_out = (float*)malloc(sizeof(float)*tri_nums);
    cudaMemcpy(v11_y_out, v11_y, sizeof(float)*tri_nums, cudaMemcpyDeviceToHost);
    cudaFree(v11_y);
    return v11_y_out;
}

__global__ void fetch_v22_y(float* v22_y, float4* tri_a_addr)
{
    for (int i = 0 ; i < tri_nums; i++){
        float4 v22 = FETCH_GLOBAL(tri_a_addr, i * 3 + 2 , float4);
        v22_y[i]  = v22.y;
    }
}

extern "C"  float* fetch_v22_y(float4* tri_a_addr)
{
    float* v22_y;
    cudaMalloc((void**)&v22_y, sizeof(float)* tri_nums);//(void**)&  其中(void **)是一个强制类型转换的。也就是把一个东西变为指向指针的指针    
    fetch_v22_y<<<1,1>>>(v22_y,tri_a_addr);
    float* v22_y_out;
    v22_y_out = (float*)malloc(sizeof(float)*tri_nums);
    cudaMemcpy(v22_y_out, v22_y, sizeof(float)*tri_nums, cudaMemcpyDeviceToHost);
    cudaFree(v22_y);
    return v22_y_out;
}

//---------------------------------------------------------------triangle_z---------------------------------------------------------
__global__ void fetch_v00_z(float* v00_z, float4* tri_a_addr)
{
    for (int i = 0 ; i < tri_nums; i++){
        float4 v00 = FETCH_GLOBAL(tri_a_addr, i * 3 , float4);
        v00_z[i]  = v00.z;
    }
}

extern "C"  float* fetch_v00_z(float4* tri_a_addr)
{
    float* v00_z;
    cudaMalloc((void**)&v00_z, sizeof(float)* tri_nums);//(void**)&  其中(void **)是一个强制类型转换的。也就是把一个东西变为指向指针的指针    
    fetch_v00_z<<<1,1>>>(v00_z,tri_a_addr);
    float* v00_z_out;
    v00_z_out = (float*)malloc(sizeof(float)*tri_nums);
    cudaMemcpy(v00_z_out, v00_z, sizeof(float)*tri_nums, cudaMemcpyDeviceToHost);
    cudaFree(v00_z);
    return v00_z_out;
}

__global__ void fetch_v11_z(float* v11_z, float4* tri_a_addr)
{
    for (int i = 0 ; i < tri_nums; i++){
        float4 v11 = FETCH_GLOBAL(tri_a_addr, i * 3 + 1 , float4);
        v11_z[i]  = v11.z;
    }
}

extern "C"  float* fetch_v11_z(float4* tri_a_addr)
{
    float* v11_z;
    cudaMalloc((void**)&v11_z, sizeof(float)* tri_nums);//(void**)&  其中(void **)是一个强制类型转换的。也就是把一个东西变为指向指针的指针    
    fetch_v11_z<<<1,1>>>(v11_z,tri_a_addr);
    float* v11_z_out;
    v11_z_out = (float*)malloc(sizeof(float)*tri_nums);
    cudaMemcpy(v11_z_out, v11_z, sizeof(float)*tri_nums, cudaMemcpyDeviceToHost);
    cudaFree(v11_z);
    return v11_z_out;
}

__global__ void fetch_v22_z(float* v22_z, float4* tri_a_addr)
{
    for (int i = 0 ; i < tri_nums; i++){
        float4 v22 = FETCH_GLOBAL(tri_a_addr, i * 3 + 2 , float4);
        v22_z[i]  = v22.z;
    }
}

extern "C"  float* fetch_v22_z(float4* tri_a_addr)
{
    float* v22_z;
    cudaMalloc((void**)&v22_z, sizeof(float)* tri_nums);//(void**)&  其中(void **)是一个强制类型转换的。也就是把一个东西变为指向指针的指针    
    fetch_v22_z<<<1,1>>>(v22_z,tri_a_addr);
    float* v22_z_out;
    v22_z_out = (float*)malloc(sizeof(float)*tri_nums);
    cudaMemcpy(v22_z_out, v22_z, sizeof(float)*tri_nums, cudaMemcpyDeviceToHost);
    cudaFree(v22_z);
    return v22_z_out;
}

//---------------------------------------------------------------triangle_w---------------------------------------------------------
__global__ void fetch_v00_w(float* v00_w, float4* tri_a_addr)
{
    for (int i = 0 ; i < tri_nums; i++){
        float4 v00 = FETCH_GLOBAL(tri_a_addr, i * 3 , float4);
        v00_w[i]  = v00.w;
    }
}

extern "C"  float* fetch_v00_w(float4* tri_a_addr)
{
    float* v00_w;
    cudaMalloc((void**)&v00_w, sizeof(float)* tri_nums);//(void**)&  其中(void **)是一个强制类型转换的。也就是把一个东西变为指向指针的指针    
    fetch_v00_w<<<1,1>>>(v00_w,tri_a_addr);
    float* v00_w_out;
    v00_w_out = (float*)malloc(sizeof(float)*tri_nums);
    cudaMemcpy(v00_w_out, v00_w, sizeof(float)*tri_nums, cudaMemcpyDeviceToHost);
    cudaFree(v00_w);
    return v00_w_out;
}

__global__ void fetch_v11_w(float* v11_w, float4* tri_a_addr)
{
    for (int i = 0 ; i < tri_nums; i++){
        float4 v11 = FETCH_GLOBAL(tri_a_addr, i * 3 + 1 , float4);
        v11_w[i]  = v11.w;
    }
}

extern "C"  float* fetch_v11_w(float4* tri_a_addr)
{
    float* v11_w;
    cudaMalloc((void**)&v11_w, sizeof(float)* tri_nums);//(void**)&  其中(void **)是一个强制类型转换的。也就是把一个东西变为指向指针的指针    
    fetch_v11_w<<<1,1>>>(v11_w,tri_a_addr);
    float* v11_w_out;
    v11_w_out = (float*)malloc(sizeof(float)*tri_nums);
    cudaMemcpy(v11_w_out, v11_w, sizeof(float)*tri_nums, cudaMemcpyDeviceToHost);
    cudaFree(v11_w);
    return v11_w_out;
}

__global__ void fetch_v22_w(float* v22_w, float4* tri_a_addr)
{
    for (int i = 0 ; i < tri_nums; i++){
        float4 v22 = FETCH_GLOBAL(tri_a_addr, i * 3 + 2 , float4);
        v22_w[i]  = v22.w;
    }
}

extern "C"  float* fetch_v22_w(float4* tri_a_addr)
{
    float* v22_w;
    cudaMalloc((void**)&v22_w, sizeof(float)* tri_nums);//(void**)&  其中(void **)是一个强制类型转换的。也就是把一个东西变为指向指针的指针    
    fetch_v22_w<<<1,1>>>(v22_w,tri_a_addr);
    float* v22_w_out;
    v22_w_out = (float*)malloc(sizeof(float)*tri_nums);
    cudaMemcpy(v22_w_out, v22_w, sizeof(float)*tri_nums, cudaMemcpyDeviceToHost);
    cudaFree(v22_w);
    return v22_w_out;
}