#include "cutil_inline_runtime.h"

#include "base/Defs.hh"
#include "base/Math.hh"
#include "Util.hh"
#include "gpu/CudaKernel.hh"
#include "kernels/CudaTracerKernels.hh"

using namespace FW;

#define node_nums 105925

__global__ void fetch_n0xy_x(float* n0xy_x, float4* node_a_addr)
{
    for (int i = 0 ; i < 5; i++){
        float4 n0xy = FETCH_GLOBAL(node_a_addr, i * 4+1, float4);
        n0xy_x[i]  = n0xy.x;
        printf("n0xy_x[%d]  = %f", i , n0xy_x[i]);
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
