#include "cutil_inline_runtime.h"

#include "base/Defs.hh"
#include "base/Math.hh"
#include "Util.hh"
#include "gpu/CudaKernel.hh"
#include "kernels/CudaTracerKernels.hh"

using namespace FW;

#define ray_nums 2073600

__global__ void fetch_origx(float* ray, float4* ray_addr)
{
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
    for (int i = 0 ; i < 2073600; i++){
        float4 o = FETCH_GLOBAL(ray_addr, i * 2 + 0, float4);
        ray[i]  = o.w;
    }
}

extern "C"  float* fetch_origw(float4* ray_addr)
{
    float* ray;
    cudaMalloc((void**)&ray, sizeof(float)* 2073600);//(void**)&  其中(void **)是一个强制类型转换的。也就是把一个东西变为指向指针的指针    
    fetch_origw<<<1,1>>>(ray,ray_addr);
    float* ray_out;
    ray_out = (float*)malloc(sizeof(float)*2073600);
    cudaMemcpy(ray_out, ray, sizeof(float)*2073600, cudaMemcpyDeviceToHost);
    cudaFree(ray);
    return ray_out;
}


__global__ void fetch_dirx(float* ray, float4* ray_addr)
{
    for (int i = 0 ; i < 2073600; i++){
        float4 d = FETCH_GLOBAL(ray_addr, i * 2 + 1, float4);
        ray[i]  = d.x;
    }
}

extern "C"  float* fetch_dirx(float4* ray_addr)
{
    float* ray;
    cudaMalloc((void**)&ray, sizeof(float)* 2073600);//(void**)&  其中(void **)是一个强制类型转换的。也就是把一个东西变为指向指针的指针    
    fetch_dirx<<<1,1>>>(ray,ray_addr);
    float* ray_out;
    ray_out = (float*)malloc(sizeof(float)*2073600);
    cudaMemcpy(ray_out, ray, sizeof(float)*2073600, cudaMemcpyDeviceToHost);
    cudaFree(ray);
    return ray_out;
}


__global__ void fetch_diry(float* ray, float4* ray_addr)
{
    for (int i = 0 ; i < 2073600; i++){
        float4 d = FETCH_GLOBAL(ray_addr, i * 2 + 1, float4);
        ray[i]  = d.y;
    }
}

extern "C"  float* fetch_diry(float4* ray_addr)
{
    float* ray;
    cudaMalloc((void**)&ray, sizeof(float)* 2073600);//(void**)&  其中(void **)是一个强制类型转换的。也就是把一个东西变为指向指针的指针    
    fetch_diry<<<1,1>>>(ray,ray_addr);
    float* ray_out;
    ray_out = (float*)malloc(sizeof(float)*2073600);
    cudaMemcpy(ray_out, ray, sizeof(float)*2073600, cudaMemcpyDeviceToHost);
    cudaFree(ray);
    return ray_out;
}


__global__ void fetch_dirz(float* ray, float4* ray_addr)
{
    for (int i = 0 ; i < 2073600; i++){
        float4 d = FETCH_GLOBAL(ray_addr, i * 2 + 1, float4);
        ray[i]  = d.z;
    }
}

extern "C"  float* fetch_dirz(float4* ray_addr)
{
    float* ray;
    cudaMalloc((void**)&ray, sizeof(float)* 2073600);//(void**)&  其中(void **)是一个强制类型转换的。也就是把一个东西变为指向指针的指针    
    fetch_dirz<<<1,1>>>(ray,ray_addr);
    float* ray_out;
    ray_out = (float*)malloc(sizeof(float)*2073600);
    cudaMemcpy(ray_out, ray, sizeof(float)*2073600, cudaMemcpyDeviceToHost);
    cudaFree(ray);
    return ray_out;
}


__global__ void fetch_dirw(float* ray, float4* ray_addr)
{
    for (int i = 0 ; i < 2073600; i++){
        float4 d = FETCH_GLOBAL(ray_addr, i * 2 + 1, float4);
        ray[i]  = d.w;
    }
}

extern "C"  float* fetch_dirw(float4* ray_addr)
{
    float* ray;
    cudaMalloc((void**)&ray, sizeof(float)* 2073600);//(void**)&  其中(void **)是一个强制类型转换的。也就是把一个东西变为指向指针的指针    
    fetch_dirw<<<1,1>>>(ray,ray_addr);
    float* ray_out;
    ray_out = (float*)malloc(sizeof(float)*2073600);
    cudaMemcpy(ray_out, ray, sizeof(float)*2073600, cudaMemcpyDeviceToHost);
    cudaFree(ray);
    return ray_out;
}



__global__ void fetch_idirx(float* ray, float4* ray_addr)
{
    float ooeps = exp2f(-80.0f); // Avoid div by zero.
    for (int i = 0 ; i < 2073600; i++){ 
        float4 d = FETCH_GLOBAL(ray_addr, i * 2 + 1, float4);
        ray[i]  = 1.0f / (fabsf(d.x) > ooeps ? d.x : copysignf(ooeps, d.x));
    }
}

extern "C"  float* fetch_idirx(float4* ray_addr)
{
    float* ray;
    cudaMalloc((void**)&ray, sizeof(float)* 2073600);//(void**)&  其中(void **)是一个强制类型转换的。也就是把一个东西变为指向指针的指针    
    fetch_idirx<<<1,1>>>(ray,ray_addr);
    float* ray_out;
    ray_out = (float*)malloc(sizeof(float)*2073600);
    cudaMemcpy(ray_out, ray, sizeof(float)*2073600, cudaMemcpyDeviceToHost);
    cudaFree(ray);
    return ray_out;
}



__global__ void fetch_idiry(float* ray, float4* ray_addr)
{
    float ooeps = exp2f(-80.0f); // Avoid div by zero.
    for (int i = 0 ; i < 2073600; i++){
        float4 d = FETCH_GLOBAL(ray_addr, i * 2 + 1, float4);
        ray[i]  = 1.0f / (fabsf(d.y) > ooeps ? d.y : copysignf(ooeps, d.y));
    }
}

extern "C"  float* fetch_idiry(float4* ray_addr)
{
    float* ray;
    cudaMalloc((void**)&ray, sizeof(float)* 2073600);//(void**)&  其中(void **)是一个强制类型转换的。也就是把一个东西变为指向指针的指针    
    fetch_idiry<<<1,1>>>(ray,ray_addr);
    float* ray_out;
    ray_out = (float*)malloc(sizeof(float)*2073600);
    cudaMemcpy(ray_out, ray, sizeof(float)*2073600, cudaMemcpyDeviceToHost);
    cudaFree(ray);
    return ray_out;
}


__global__ void fetch_idirz(float* ray, float4* ray_addr)
{
    float ooeps = exp2f(-80.0f); // Avoid div by zero.
    for (int i = 0 ; i < 2073600; i++){
        float4 d = FETCH_GLOBAL(ray_addr, i * 2 + 1, float4);
        ray[i]  = 1.0f / (fabsf(d.z) > ooeps ? d.z : copysignf(ooeps, d.z));
    }
}

extern "C"  float* fetch_idirz(float4* ray_addr)
{
    float* ray;
    cudaMalloc((void**)&ray, sizeof(float)* 2073600);//(void**)&  其中(void **)是一个强制类型转换的。也就是把一个东西变为指向指针的指针    
    fetch_idirz<<<1,1>>>(ray,ray_addr);
    float* ray_out;
    ray_out = (float*)malloc(sizeof(float)*2073600);
    cudaMemcpy(ray_out, ray, sizeof(float)*2073600, cudaMemcpyDeviceToHost);
    cudaFree(ray);
    return ray_out;
}


__global__ void fetch_oodx(float* ray, float4* ray_addr)
{
    float ooeps = exp2f(-80.0f); // Avoid div by zero.
    float idirx;
    for (int i = 0 ; i < 2073600; i++){
        float4 o = FETCH_GLOBAL(ray_addr, i * 2 + 0, float4);
        float4 d = FETCH_GLOBAL(ray_addr, i * 2 + 1, float4);
        idirx    = 1.0f / (fabsf(d.x) > ooeps ? d.x : copysignf(ooeps, d.x));
        ray[i]  = idirx * o.x;
    }
}

extern "C"  float* fetch_oodx(float4* ray_addr)
{
    float* ray;
    cudaMalloc((void**)&ray, sizeof(float)* 2073600);//(void**)&  其中(void **)是一个强制类型转换的。也就是把一个东西变为指向指针的指针    
    fetch_oodx<<<1,1>>>(ray,ray_addr);
    float* ray_out;
    ray_out = (float*)malloc(sizeof(float)*2073600);
    cudaMemcpy(ray_out, ray, sizeof(float)*2073600, cudaMemcpyDeviceToHost);
    cudaFree(ray);
    return ray_out;
}


__global__ void fetch_oody(float* ray, float4* ray_addr)
{
    float ooeps = exp2f(-80.0f); // Avoid div by zero.
    float idiry;
    for (int i = 0 ; i < 2073600; i++){
        float4 o = FETCH_GLOBAL(ray_addr, i * 2 + 0, float4);
        float4 d = FETCH_GLOBAL(ray_addr, i * 2 + 1, float4);
        idiry    = 1.0f / (fabsf(d.y) > ooeps ? d.y : copysignf(ooeps, d.y));
        ray[i]  = idiry * o.y;
    }
}

extern "C"  float* fetch_oody(float4* ray_addr)
{
    float* ray;
    cudaMalloc((void**)&ray, sizeof(float)* 2073600);//(void**)&  其中(void **)是一个强制类型转换的。也就是把一个东西变为指向指针的指针    
    fetch_oody<<<1,1>>>(ray,ray_addr);
    float* ray_out;
    ray_out = (float*)malloc(sizeof(float)*2073600);
    cudaMemcpy(ray_out, ray, sizeof(float)*2073600, cudaMemcpyDeviceToHost);
    cudaFree(ray);
    return ray_out;
}

__global__ void fetch_oodz(float* ray, float4* ray_addr)
{
    float ooeps = exp2f(-80.0f); // Avoid div by zero.
    float idirz;
    for (int i = 0 ; i < 2073600; i++){
        float4 o = FETCH_GLOBAL(ray_addr, i * 2 + 0, float4);
        float4 d = FETCH_GLOBAL(ray_addr, i * 2 + 1, float4);
        idirz    = 1.0f / (fabsf(d.z) > ooeps ? d.z : copysignf(ooeps, d.z));
        ray[i]  = idirz * o.z;
    }
}

extern "C"  float* fetch_oodz(float4* ray_addr)
{
    float* ray;
    cudaMalloc((void**)&ray, sizeof(float)* 2073600);//(void**)&  其中(void **)是一个强制类型转换的。也就是把一个东西变为指向指针的指针    
    fetch_oodz<<<1,1>>>(ray,ray_addr);
    float* ray_out;
    ray_out = (float*)malloc(sizeof(float)*2073600);
    cudaMemcpy(ray_out, ray, sizeof(float)*2073600, cudaMemcpyDeviceToHost);
    cudaFree(ray);
    return ray_out;
}