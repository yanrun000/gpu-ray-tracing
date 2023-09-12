#include "cutil_inline_runtime.h"

#include "base/Defs.hh"
#include "base/Math.hh"
#include "Util.hh"
#include "gpu/CudaKernel.hh"
#include "CudaTracerKernels.hh"

using namespace FW;


__global__ float4* fetch_rays(float4* ray_s)
{

    static float4 ray_0[1];


    for(int i = 0 ; i < 1; i++)
    {
        ray_0[i] = FETCH_GLOBAL(ray_s, i * 2 + 0, float4);
    }
  
    return ray_0;

}