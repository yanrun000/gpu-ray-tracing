/*
 *  Copyright (c) 2009-2011, NVIDIA Corporation
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *      * Redistributions of source code must retain the above copyright
 *        notice, this list of conditions and the following disclaimer.
 *      * Redistributions in binary form must reproduce the above copyright
 *        notice, this list of conditions and the following disclaimer in the
 *        documentation and/or other materials provided with the distribution.
 *      * Neither the name of NVIDIA Corporation nor the
 *        names of its contributors may be used to endorse or promote products
 *        derived from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 *  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 *  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *  DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
 *  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 *  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 *  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 *  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef CUDATRACERKERNELS_HH
#define CUDATRACERKERNELS_HH

//#include <cuda.h>
#include "base/Defs.hh"
#include "base/Math.hh"
#include "Util.hh"
#include "gpu/CudaKernel.hh"
//------------------------------------------------------------------------
// Constants.
//------------------------------------------------------------------------
namespace FW
{

extern "C"
{
    void bind_CudaBVHTexture(float4*, S64, float4*, S64, int*, S64);
    
    void unbind_CudaBVHTexture(void);
    
    float launch_tracingKernel(S32 threads, Vec2i& blockSize, int numRays, bool anyHit, float4* rays, int4* results, 
                               float4* nodesA, float4* nodesB, float4* nodesC, float4* nodesD, 
                               float4* trisA, float4* trisB, float4* trisC, int* triIndices);
                                     
    void copy_tracing_results(RayResult* result_host, RayResult* result_dev, S32 size);
}

enum
{
    MaxBlockHeight      = 6,            // Upper bound for blockDim.y.
    EntrypointSentinel  = 0x76543210,   // Bottom-most stack entry, indicating the end of traversal.
};

//------------------------------------------------------------------------
// BVH memory layout.
//------------------------------------------------------------------------

enum BVHLayout
{
    BVHLayout_AOS_AOS = 0,              // Nodes = array-of-structures, triangles = array-of-structures. Used by tesla_xxx kernels.
    BVHLayout_AOS_SOA,                  // Nodes = array-of-structures, triangles = structure-of-arrays.
    BVHLayout_SOA_AOS,                  // Nodes = structure-of-arrays, triangles = array-of-structures.
    BVHLayout_SOA_SOA,                  // Nodes = structure-of-arrays, triangles = structure-of-arrays.
    BVHLayout_Compact,                  // Variant of BVHLayout_AOS_AOS with implicit leaf nodes.
    BVHLayout_Compact2,                 // Variant of BVHLayout_AOS_AOS with implicit leaf nodes.

    BVHLayout_Max
};

//------------------------------------------------------------------------
// Kernel configuration. Written by queryConfig() in each CU file.
//------------------------------------------------------------------------

struct KernelConfig
{
    int         bvhLayout;              // Desired BVHLayout.
    int         blockWidth;             // Desired blockDim.x.
    int         blockHeight;            // Desired blockDim.y.
    int         usePersistentThreads;   // True to enable persistent threads.
};

//------------------------------------------------------------------------
// Function signature for trace().
//------------------------------------------------------------------------



//__global__ void texture_test(void);

//------------------------------------------------------------------------
// Temporary data stored in shared memory to reduce register pressure.
//------------------------------------------------------------------------

struct RayStruct
{
    float   idirx;  // 1.0f / ray.direction.x
    float   idiry;  // 1.0f / ray.direction.y
    float   idirz;  // 1.0f / ray.direction.z
    float   tmin;   // ray.tmin
    float   dummy;  // Padding to avoid bank conflicts.
};

//------------------------------------------------------------------------
// Globals.
//------------------------------------------------------------------------

//#ifdef __CUDACC__
#if FW_CUDA
extern "C"
{
#define TRACE_FUNC \
    __global__ void trace( \
        int             numRays,        /* Total number of rays in the batch. */ \
        bool            anyHit,         /* False if rays need to find the closest hit. */ \
        float4*         rays,           /* Ray input: float3 origin, float tmin, float3 direction, float tmax. */ \
        int4*           results,        /* Ray output: int triangleID, float hitT, int2 padding. */ \
        float4*         nodesA,         /* SOA: bytes 0-15 of each node, AOS/Compact: 64 bytes per node. */ \
        float4*         nodesB,         /* SOA: bytes 16-31 of each node, AOS/Compact: unused. */ \
        float4*         nodesC,         /* SOA: bytes 32-47 of each node, AOS/Compact: unused. */ \
        float4*         nodesD,         /* SOA: bytes 48-63 of each node, AOS/Compact: unused. */ \
        float4*         trisA,          /* SOA: bytes 0-15 of each triangle, AOS: 64 bytes per triangle, Compact: 48 bytes per triangle. */ \
        float4*         trisB,          /* SOA: bytes 16-31 of each triangle, AOS/Compact: unused. */ \
        float4*         trisC,          /* SOA: bytes 32-47 of each triangle, AOS/Compact: unused. */ \
        int*            triIndices)     /* Triangle index remapping table. */

texture<float4, 1> t_nodesA;
texture<float4, 1> t_trisA;
texture<int,  1>   t_triIndices;
}
#endif

//------------------------------------------------------------------------
// Utilities.
//------------------------------------------------------------------------

#define FETCH_GLOBAL(NAME, IDX, TYPE) ((const TYPE*)NAME)[IDX]
#define FETCH_TEXTURE(NAME, IDX, TYPE) tex1Dfetch(t_ ## NAME, IDX)
#define STORE_RESULT(RAY, TRI, T) ((int2*)results)[(RAY) * 2] = make_int2(TRI, __float_as_int(T))

//------------------------------------------------------------------------

#ifdef __CUDACC__

/*template <class T> __device__ __inline__ void swap(T& a,T& b)
{
    T t = a;
    a = b;
    b = t;
}*/

__device__ __inline__ float min4(float a, float b, float c, float d)
{
    return fminf(fminf(fminf(a, b), c), d);
}

__device__ __inline__ float max4(float a, float b, float c, float d)
{
    return fmaxf(fmaxf(fmaxf(a, b), c), d);
}

__device__ __inline__ float min3(float a, float b, float c)
{
    return fminf(fminf(a, b), c);
}

__device__ __inline__ float max3(float a, float b, float c)
{
    return fmaxf(fmaxf(a, b), c);
}

// Using integer min,max
__inline__ __device__ float fminf2(float a,float b)
{
    int a2 = __float_as_int(a);
    int b2 = __float_as_int(b);
    return __int_as_float( a2<b2 ? a2 : b2 );
}

__inline__ __device__ float fmaxf2(float a,float b)
{
    int a2 = __float_as_int(a);
    int b2 = __float_as_int(b);
    return __int_as_float( a2>b2 ? a2 : b2 );
}


// Using video instructions
__device__ __inline__ int   min_min   (int a, int b, int c) { int v; asm("vmin.s32.s32.s32.min %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ int   min_max   (int a, int b, int c) { int v; asm("vmin.s32.s32.s32.max %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ int   max_min   (int a, int b, int c) { int v; asm("vmax.s32.s32.s32.min %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ int   max_max   (int a, int b, int c) { int v; asm("vmax.s32.s32.s32.max %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ float fmin_fmin (float a, float b, float c) { return __int_as_float(min_min(__float_as_int(a), __float_as_int(b), __float_as_int(c))); }
__device__ __inline__ float fmin_fmax (float a, float b, float c) { return __int_as_float(min_max(__float_as_int(a), __float_as_int(b), __float_as_int(c))); }
__device__ __inline__ float fmax_fmin (float a, float b, float c) { return __int_as_float(max_min(__float_as_int(a), __float_as_int(b), __float_as_int(c))); }
__device__ __inline__ float fmax_fmax (float a, float b, float c) { return __int_as_float(max_max(__float_as_int(a), __float_as_int(b), __float_as_int(c))); }


__device__ __inline__ float magic_max7(float a0, float a1, float b0, float b1, float c0, float c1, float d)
{
            float t1 = fmin_fmax(a0, a1, d);
            float t2 = fmin_fmax(b0, b1, t1);
            float t3 = fmin_fmax(c0, c1, t2);
            return t3;
}

__device__ __inline__ float magic_min7(float a0, float a1, float b0, float b1, float c0, float c1, float d)
{
            float t1 = fmax_fmin(a0, a1, d);
            float t2 = fmax_fmin(b0, b1, t1);
            float t3 = fmax_fmin(c0, c1, t2);
            return t3;
}

// Experimentally determined best mix of float/int/video minmax instructions for Kepler.
__device__ __inline__ float spanBeginKepler(float a0, float a1, float b0, float b1, float c0, float c1, float d){    return fmax_fmax( fminf(a0,a1), fminf(b0,b1), fmin_fmax(c0, c1, d)); }
__device__ __inline__ float spanEndKepler(float a0, float a1, float b0, float b1, float c0, float c1, float d)    {    return fmin_fmin( fmaxf(a0,a1), fmaxf(b0,b1), fmax_fmin(c0, c1, d)); }

// Same for Fermi.
__device__ __inline__ float spanBeginFermi(float a0, float a1, float b0, float b1, float c0, float c1, float d) {    return magic_max7(a0, a1, b0, b1, c0, c1, d); }
__device__ __inline__ float spanEndFermi(float a0, float a1, float b0, float b1, float c0, float c1, float d)    {    return magic_min7(a0, a1, b0, b1, c0, c1, d); }


#endif

//------------------------------------------------------------------------
}
#endif //CUDATRACERKERNELS_HH