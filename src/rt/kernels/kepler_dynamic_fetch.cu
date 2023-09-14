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

/*
    GK104-optimized variant of the "Persistent speculative
    while-while" kernel used in:

    "Understanding the Efficiency of Ray Traversal on GPUs",
    Timo Aila and Samuli Laine,
    Proc. High-Performance Graphics 2009

    This variant fetches new work dynamically as soon as the
    warp occupancy drops below a pre-determined threshold.
*/

#include "CudaTracerKernels.hh"
#include "base/Defs.hh"
//#include <cutil_inline.h>
#include "cutil_inline_runtime.h"

//------------------------------------------------------------------------

#define STACK_SIZE              64          // Size of the traversal stack in local memory.
#define DYNAMIC_FETCH_THRESHOLD 20          // If fewer than this active, fetch new rays

using namespace FW;

__device__ int g_warpCounter[1];    // Work counter for persistent threads.

//------------------------------------------------------------------------

/*extern "C" __global__ void queryConfig(void)
{
    g_config.bvhLayout = BVHLayout_Compact2;
    g_config.blockWidth = 32;
    g_config.blockHeight = 4;
    g_config.usePersistentThreads = 1;
}*/

//------------------------------------------------------------------------

TRACE_FUNC
{
    // Traversal stack in CUDA thread-local memory.

    int traversalStack[STACK_SIZE];
    traversalStack[0] = EntrypointSentinel; // Bottom-most entry.

    // Live state during traversal, stored in registers.

    float   origx, origy, origz;            // Ray origin.
    char*   stackPtr;                       // Current position in traversal stack.
    int     leafAddr;                       // First postponed leaf, non-negative if none.
    int     leafAddr2;                      // Second postponed leaf, non-negative if none.
    int     nodeAddr = EntrypointSentinel;  // Non-negative: current internal node, negative: second postponed leaf.
    int     hitIndex;                       // Triangle index of the closest intersection, -1 if none.
    float   hitT;                           // t-value of the closest intersection.
    float   tmin;
    int     rayidx;
    float   oodx;
    float   oody;
    float   oodz;
    float   dirx;
    float   diry;
    float   dirz;
    float   idirx;
    float   idiry;
    float   idirz;

    // Initialize persistent threads.

    __shared__ volatile int nextRayArray[MaxBlockHeight]; // Current ray index in global buffer.

    // Persistent threads: fetch and process rays in a loop.

    do
    {
        const int tidx = threadIdx.x;
        volatile int& rayBase = nextRayArray[threadIdx.y];

        // Fetch new rays from the global pool using lane 0.

        const bool          terminated     = nodeAddr==EntrypointSentinel;
        const unsigned int  maskTerminated = __ballot(terminated);
        const int           numTerminated  = __popc(maskTerminated);
        const int           idxTerminated  = __popc(maskTerminated & ((1u<<tidx)-1));

        if(terminated)
        {
            if (idxTerminated == 0)
                rayBase = atomicAdd(&g_warpCounter[0], numTerminated);

            rayidx = rayBase + idxTerminated;
            if (rayidx >= numRays)
                break;

            // Fetch ray.

            float4 o = FETCH_GLOBAL(rays, rayidx * 2 + 0, float4);
            float4 d = FETCH_GLOBAL(rays, rayidx * 2 + 1, float4);
            origx = o.x;  
            origy = o.y;
            origz = o.z;
            tmin  = o.w;
            dirx  = d.x;
            diry  = d.y;
            dirz  = d.z;
            hitT  = d.w;

            // int ray_id = 0;
            // // if(Second_Ray){
            // for(int ray_id = 0; ray_id <= 4 ; ray_id ++)
            // {
            //     float4 oa = FETCH_GLOBAL(rays, ray_id * 2 + 0, float4);
            //     printf("origx!!!! = %x\n", __float_as_int(oa.x));
                // printf("hello world!  %d\n",ray_id);

            // }
            // }


            float ooeps = exp2f(-80.0f); // Avoid div by zero.
            idirx = 1.0f / (fabsf(d.x) > ooeps ? d.x : copysignf(ooeps, d.x));
            idiry = 1.0f / (fabsf(d.y) > ooeps ? d.y : copysignf(ooeps, d.y));
            idirz = 1.0f / (fabsf(d.z) > ooeps ? d.z : copysignf(ooeps, d.z));
            oodx  = origx * idirx;
            oody  = origy * idiry;
            oodz  = origz * idirz;

            // Setup traversal.

            stackPtr = (char*)&traversalStack[0];
            leafAddr = 0;   // No postponed leaf.
            leafAddr2= 0;   // No postponed leaf.
            nodeAddr = 0;   // Start from the root.
            hitIndex = -1;  // No triangle intersected so far.
        }

        // Traversal loop.

        while(nodeAddr != EntrypointSentinel)
        {
            // Traverse internal nodes until all SIMD lanes have found a leaf.

//          while (nodeAddr >= 0 && nodeAddr != EntrypointSentinel)
            while (((unsigned int) nodeAddr) < ((unsigned int) EntrypointSentinel))   // functionally equivalent, but faster
            {
                // Fetch AABBs of the two child nodes.

                const float4 n0xy = tex1Dfetch(t_nodesA, nodeAddr + 0); // (c0.lo.x, c0.hi.x, c0.lo.y, c0.hi.y)
                const float4 n1xy = tex1Dfetch(t_nodesA, nodeAddr + 1); // (c1.lo.x, c1.hi.x, c1.lo.y, c1.hi.y)
                const float4 nz   = tex1Dfetch(t_nodesA, nodeAddr + 2); // (c0.lo.z, c0.hi.z, c1.lo.z, c1.hi.z)
                      float4 tmp  = tex1Dfetch(t_nodesA, nodeAddr + 3); // child_index0, child_index1
                      int2  cnodes= *(int2*)&tmp;

                // Intersect the ray against the child nodes.

                const float c0lox = n0xy.x * idirx - oodx;
                const float c0hix = n0xy.y * idirx - oodx;
                const float c0loy = n0xy.z * idiry - oody;
                const float c0hiy = n0xy.w * idiry - oody;
                const float c0loz = nz.x   * idirz - oodz;
                const float c0hiz = nz.y   * idirz - oodz;
                const float c1loz = nz.z   * idirz - oodz;
                const float c1hiz = nz.w   * idirz - oodz;
                const float c0min = spanBeginKepler(c0lox, c0hix, c0loy, c0hiy, c0loz, c0hiz, tmin);
                const float c0max = spanEndKepler  (c0lox, c0hix, c0loy, c0hiy, c0loz, c0hiz, hitT);
                //float c0min = max4(fminf(c0lox, c0hix), fminf(c0loy, c0hiy), fminf(c0loz, c0hiz), tmin);
                //float c0max = min4(fmaxf(c0lox, c0hix), fmaxf(c0loy, c0hiy), fmaxf(c0loz, c0hiz), hitT);
                
                const float c1lox = n1xy.x * idirx - oodx;
                const float c1hix = n1xy.y * idirx - oodx;
                const float c1loy = n1xy.z * idiry - oody;
                const float c1hiy = n1xy.w * idiry - oody;
                const float c1min = spanBeginKepler(c1lox, c1hix, c1loy, c1hiy, c1loz, c1hiz, tmin);
                const float c1max = spanEndKepler  (c1lox, c1hix, c1loy, c1hiy, c1loz, c1hiz, hitT);
                //float c1min = max4(fminf(c1lox, c1hix), fminf(c1loy, c1hiy), fminf(c1loz, c1hiz), tmin);
                //float c1max = min4(fmaxf(c1lox, c1hix), fmaxf(c1loy, c1hiy), fmaxf(c1loz, c1hiz), hitT);


                bool swp = (c1min < c0min);

                bool traverseChild0 = (c0max >= c0min);
                bool traverseChild1 = (c1max >= c1min);

                // Neither child was intersected => pop stack.

                if (!traverseChild0 && !traverseChild1)
                {
                    nodeAddr = *(int*)stackPtr;
                    stackPtr -= 4;
                }

                // Otherwise => fetch child pointers.

                else
                {
                    nodeAddr = (traverseChild0) ? cnodes.x : cnodes.y;

                    // Both children were intersected => push the farther one.

                    if (traverseChild0 && traverseChild1)
                    {
                        if (swp)
                            swap(nodeAddr, cnodes.y);
                        stackPtr += 4;
                        *(int*)stackPtr = cnodes.y;
                    }
                }

                // First leaf => postpone and continue traversal.

                if (nodeAddr < 0 && leafAddr  >= 0)     // Postpone max 1
//              if (nodeAddr < 0 && leafAddr2 >= 0)     // Postpone max 2
                {
                    //leafAddr2= leafAddr;          // postpone 2
                    leafAddr = nodeAddr;
                    nodeAddr = *(int*)stackPtr;
                    stackPtr -= 4;
                }

                // All SIMD lanes have found a leaf? => process them.

                // NOTE: inline PTX implementation of "if(!__any(leafAddr >= 0)) break;".
                // tried everything with CUDA 4.2 but always got several redundant instructions.

                unsigned int mask;
                asm("{\n"
                    "   .reg .pred p;               \n"
                    "setp.ge.s32        p, %1, 0;   \n"
                    "vote.ballot.b32    %0,p;       \n"
                    "}"
                    : "=r"(mask)
                    : "r"(leafAddr));
                if(!mask)
                    break;

                //if(!__any(leafAddr >= 0))
                //    break;
            }

            // Process postponed leaf nodes.

            while (leafAddr < 0)
            {
                for (int triAddr = ~leafAddr;; triAddr += 3)
                {
                    // Tris in TEX (good to fetch as a single batch)
                    const float4 v00 = tex1Dfetch(t_trisA, triAddr + 0);
                    const float4 v11 = tex1Dfetch(t_trisA, triAddr + 1);
                    const float4 v22 = tex1Dfetch(t_trisA, triAddr + 2);

                    // End marker (negative zero) => all triangles processed.
                    if (__float_as_int(v00.x) == 0x80000000)
                        break;

                    float Oz = v00.w - origx*v00.x - origy*v00.y - origz*v00.z;
                    float invDz = 1.0f / (dirx*v00.x + diry*v00.y + dirz*v00.z);
                    float t = Oz * invDz;

                    if (t > tmin && t < hitT)
                    {
                        // Compute and check barycentric u.

                        float Ox = v11.w + origx*v11.x + origy*v11.y + origz*v11.z;
                        float Dx = dirx*v11.x + diry*v11.y + dirz*v11.z;
                        float u = Ox + t*Dx;

                        if (u >= 0.0f)
                        {
                            // Compute and check barycentric v.

                            float Oy = v22.w + origx*v22.x + origy*v22.y + origz*v22.z;
                            float Dy = dirx*v22.x + diry*v22.y + dirz*v22.z;
                            float v = Oy + t*Dy;

                            if (v >= 0.0f && u + v <= 1.0f)
                            {
                                // Record intersection.
                                // Closest intersection not required => terminate.

                                hitT = t;
                                hitIndex = triAddr;
                                if (anyHit)
                                {
                                    nodeAddr = EntrypointSentinel;
                                    break;
                                }
                            }
                        }
                    }
                } // triangle

                // Another leaf was postponed => process it as well.

//              if(leafAddr2<0) { leafAddr = leafAddr2; leafAddr2=0; } else     // postpone2
                {
                    leafAddr = nodeAddr;
                    if (nodeAddr < 0)
                    {
                        nodeAddr = *(int*)stackPtr;
                        stackPtr -= 4;
                    }
                }
            } // leaf

            // DYNAMIC FETCH

            if( __popc(__ballot(true)) < DYNAMIC_FETCH_THRESHOLD )
                break;

        } // traversal

        // Remap intersected triangle index, and store the result.

        if (hitIndex == -1) { STORE_RESULT(rayidx, -1, hitT); }
        else                { STORE_RESULT(rayidx, FETCH_TEXTURE(triIndices, hitIndex, int), hitT); }

    } while(true);
}



//------------------------------------------------------------------------

extern "C" void bind_CudaBVHTexture(float4* nodeBuf, S64 nodeBufSize, float4* triWoopBuf, S64 triWoopSize, int* triIndexBuf, S64 triIndexSize)
{
    //printf("triWoopSize = %d, triIndexSize = %d\n", triWoopSize, triIndexSize);
    
    cutilSafeCall(cudaBindTexture(NULL, t_nodesA, nodeBuf, cudaCreateChannelDesc<float4>(), nodeBufSize));
    cutilSafeCall(cudaBindTexture(NULL, t_trisA, triWoopBuf, cudaCreateChannelDesc<float4>(), triWoopSize));
    cutilSafeCall(cudaBindTexture(NULL, t_triIndices, triIndexBuf, cudaCreateChannelDesc<int>(), triIndexSize));
}

extern "C" void unbind_CudaBVHTexture(void)
{
    cutilSafeCall(cudaUnbindTexture(t_trisA));
    cutilSafeCall(cudaUnbindTexture(t_triIndices));
}

extern "C" float launch_tracingKernel(S32 nthreads, Vec2i& blockSize, int numRays, bool anyHit, float4* rays, int4* results, 
                                      float4* nodesA, float4* nodesB, float4* nodesC, float4* nodesD, float4* trisA, float4* trisB, float4* trisC, int* triIndices)
{
    class CudaKernel cukernel;
    cukernel.setGrid(nthreads, blockSize);
    
    dim3 dimBlock(cukernel.getBlockSize().x, cukernel.getBlockSize().y, 1);
    dim3 dimGrid(cukernel.getGridSize().x, cukernel.getGridSize().y, 1);
    
    int warpCounter[1];
    
    warpCounter[0] = 0;
    
    cutilSafeCall(cudaMemcpyToSymbol(g_warpCounter, warpCounter, sizeof(int)));
    
    printf("launching tracingKernel... Block(%d, %d), Grid(%d, %d)\n", cukernel.getBlockSize().x, cukernel.getBlockSize().y, cukernel.getGridSize().x, cukernel.getGridSize().y);
    
    cudaEvent_t start, stop;
    
    cutilSafeCall(cudaEventCreate(&start));
    cutilSafeCall(cudaEventCreate(&stop));
    cutilSafeCall(cudaEventRecord(start, 0));
    
    trace <<<dimGrid, dimBlock>>> (numRays, anyHit, rays, results, nodesA, nodesB, nodesC, nodesD, trisA, trisB, trisC, triIndices);
    
    // float4 ray_store[0];
    // int ray_id = 0
    // cudaMemcpy(ray_store[0], FETCH_GLOBAL(rays, 0, float4), sizeof(float4)* 1, cudaMemcpyDeviceToHost);



    // cudaFree(ray_store);

    cutilSafeCall(cudaEventRecord(stop, 0));
    cutilSafeCall(cudaEventSynchronize(stop));
    
    float elapsedTime;
    
    cutilSafeCall(cudaEventElapsedTime(&elapsedTime, start, stop));
    
    return elapsedTime;
}

extern "C" void copy_tracing_results(RayResult* result_host, RayResult* result_dev, S32 size)
{
    cutilSafeCall(cudaMemcpy(result_host, result_dev, size * sizeof(RayResult), cudaMemcpyDeviceToHost));
}
