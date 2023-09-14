#include "cutil_inline_runtime.h"

#include "base/Defs.hh"
#include "base/Math.hh"
#include "Util.hh"
#include "gpu/CudaKernel.hh"
#include "kernels/CudaTracerKernels.hh"

using namespace FW;

__global__ void fetch_origx(float* ray, float4* ray_addr)
{
    // static float ray[1];
    for (int i = 0 ; i < 2073600; i++){
        float4 o = FETCH_GLOBAL(ray_addr, i * 2 + 0, float4);
        ray[i]  = o.x;
    }
}

extern "C"  float* fetch_origx(float4* ray_addr)
{
    float* ray;
    cudaMalloc((void**)&ray, sizeof(float)* 2073600);//(void**)&  其中(void **)是一个强制类型转换的。也就是把一个东西变为指向指针的指针    
    fetch_origx<<<1,1>>>(ray,ray_addr);
    float* ray_out;
    ray_out = (float*)malloc(sizeof(float)*2073600);
    cudaMemcpy(ray_out, ray, sizeof(float)*2073600, cudaMemcpyDeviceToHost);
    cudaFree(ray);
    return ray_out;
}


__global__ void fetch_origy(float* ray, float4* ray_addr)
{
    // static float ray[1];
    for (int i = 0 ; i < 2073600; i++){
        float4 o = FETCH_GLOBAL(ray_addr, i * 2 + 0, float4);
        ray[i]  = o.y;
    }
}

extern "C"  float* fetch_origy(float4* ray_addr)
{
    float* ray;
    cudaMalloc((void**)&ray, sizeof(float)* 2073600);//(void**)&  其中(void **)是一个强制类型转换的。也就是把一个东西变为指向指针的指针    
    fetch_origy<<<1,1>>>(ray,ray_addr);
    float* ray_out;
    ray_out = (float*)malloc(sizeof(float)*2073600);
    cudaMemcpy(ray_out, ray, sizeof(float)*2073600, cudaMemcpyDeviceToHost);
    cudaFree(ray);
    return ray_out;
}


__global__ void fetch_origz(float* ray, float4* ray_addr)
{
    // static float ray[1];
    for (int i = 0 ; i < 2073600; i++){
        float4 o = FETCH_GLOBAL(ray_addr, i * 2 + 0, float4);
        ray[i]  = o.z;
    }
}

extern "C"  float* fetch_origz(float4* ray_addr)
{
    float* ray;
    cudaMalloc((void**)&ray, sizeof(float)* 2073600);//(void**)&  其中(void **)是一个强制类型转换的。也就是把一个东西变为指向指针的指针    
    fetch_origz<<<1,1>>>(ray,ray_addr);
    float* ray_out;
    ray_out = (float*)malloc(sizeof(float)*2073600);
    cudaMemcpy(ray_out, ray, sizeof(float)*2073600, cudaMemcpyDeviceToHost);
    cudaFree(ray);
    return ray_out;
}


__global__ void fetch_origw(float* ray, float4* ray_addr)
{
    // static float ray[1];
    for (int i = 0 ; i < 2073600; i++){
        float4 o = FETCH_GLOBAL(ray_addr, i * 2 + 0, float4);
        ray[i]  = o.z;
    }
}

extern "C"  float* fetch_origw(float4* ray_addr)
{
    float* ray;
    cudaMalloc((void**)&ray, sizeof(float)* 2073600);//(void**)&  其中(void **)是一个强制类型转换的。也就是把一个东西变为指向指针的指针    
    fetch_origz<<<1,1>>>(ray,ray_addr);
    float* ray_out;
    ray_out = (float*)malloc(sizeof(float)*2073600);
    cudaMemcpy(ray_out, ray, sizeof(float)*2073600, cudaMemcpyDeviceToHost);
    cudaFree(ray);
    return ray_out;
}
