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

//#include <cutil_inline.h>
#include "cutil_inline_runtime.h"

#include "cuda/CudaTracer.hh"
#include "gpu/CudaModule.hh"
#include "kernels/CudaTracerKernels.hh"

#include<iostream>

//#define DUMP_RAYS

//#include "gui/Window.hpp"
//#include "io/File.hh"

using namespace FW;

bool        CudaTracer::s_inited        = false;

//------------------------------------------------------------------------

CudaTracer::CudaTracer(void)
:   m_bvh(NULL)
{
    CudaModule::staticInit();
    //m_compiler.addOptions("-use_fast_math");
    IsTexMemBinded = false;
}

//------------------------------------------------------------------------

CudaTracer::~CudaTracer(void)
{
    if(IsTexMemBinded)
        unbind_CudaBVHTexture();
}

//------------------------------------------------------------------------
void CudaTracer::initKernel(void)
{
    // Not changed => done.
    
    if (s_inited) return;
    
    s_inited = true;
    
    //Configuration for fermi_speculative_while_while
    //m_kernelConfig.bvhLayout = BVHLayout_Compact;
    //m_kernelConfig.blockWidth = 32;
    //m_kernelConfig.blockHeight = 6;
    //m_kernelConfig.usePersistentThreads = 0;
    
    //Configuration for fermi_dynamic_fetch
    //m_kernelConfig.bvhLayout = BVHLayout_Compact;
    //m_kernelConfig.blockWidth = 32;
    //m_kernelConfig.blockHeight = 6;
    //m_kernelConfig.usePersistentThreads = 1;    
    
    //Configuration for kepler_dynamic_fetch
    m_kernelConfig.bvhLayout = BVHLayout_Compact2;
    m_kernelConfig.blockWidth = 32;
    m_kernelConfig.blockHeight = 4;
    m_kernelConfig.usePersistentThreads = 1;
    
    //Configuration for kepler_speculative_while_while
    //m_kernelConfig.bvhLayout = BVHLayout_Compact2;
    //m_kernelConfig.blockWidth = 32;
    //m_kernelConfig.blockHeight = 4;
    //m_kernelConfig.usePersistentThreads = 0;
}

//------------------------------------------------------------------------

F32 CudaTracer::traceBatch(RayBuffer& rays)
{
    // No rays => done.
    // Second_Ray = true;
    int numRays = rays.getSize();
    if (!numRays)
        return 0.0f;

    // Check BVH consistency.

    if (!m_bvh)
        fail("CudaTracer: No BVH!");
    if (m_bvh->getLayout() != getDesiredBVHLayout())
        fail("CudaTracer: Incorrect BVH layout!");
    
    Vec2i   nodeOfsA    = m_bvh->getNodeSubArray(0);
    Vec2i   nodeOfsB    = m_bvh->getNodeSubArray(1);
    Vec2i   nodeOfsC    = m_bvh->getNodeSubArray(2);
    Vec2i   nodeOfsD    = m_bvh->getNodeSubArray(3);
    Vec2i   triOfsA     = m_bvh->getTriWoopSubArray(0);
    Vec2i   triOfsB     = m_bvh->getTriWoopSubArray(1);
    Vec2i   triOfsC     = m_bvh->getTriWoopSubArray(2);
    
    if(IsTexMemBinded == false)
    {
        bind_CudaBVHTexture(m_bvh->getNodeBuffer_dev(), nodeOfsA.y, m_bvh->getTriWoopBuffer_dev(), triOfsA.y, m_bvh->getTriIndexBuffer_dev(), m_bvh->getTriIndexBuffer().getSize());
        IsTexMemBinded = true;
    }
    
#if defined(DUMP_RAYS)
    rays.dumpRayBuffer();
#endif
    
    int desiredWarps = (numRays + 31) / 32;
    if (m_kernelConfig.usePersistentThreads != 0)
    {
        //*(S32*)module->getGlobal("g_warpCounter").getMutablePtr() = 0;
        desiredWarps = 720; // Tesla: 30 SMs * 24 warps, Fermi: 15 SMs * 48 warps
    }

    Vec2i blockSize(m_kernelConfig.blockWidth, m_kernelConfig.blockHeight);
    int blockWarps = (blockSize.x * blockSize.y + 31) / 32;
    int numBlocks = (desiredWarps + blockWarps - 1) / blockWarps;
    
    printf("Ray Size = %d ", numRays);
    
    if(rays.getNeedClosestHit()) printf("Any Hit: FALSE\n");
    else printf("Any Hit: TRUE\n");

   
    // printf("rays.x = %f\n",(float4*) rays.getRayBuffer_dev());
    // bool second_ray = m_second_ray;
    float trace_time = launch_tracingKernel(numBlocks * blockSize.x * blockSize.y, blockSize, numRays, (rays.getNeedClosestHit()) ? 0 : 1, (float4*) rays.getRayBuffer_dev(),
                         (int4*) rays.getResultBuffer_dev(), 
                         (float4*) (m_bvh->getNodeBuffer_dev() + nodeOfsA.x), (float4*) (m_bvh->getNodeBuffer_dev() + nodeOfsB.x),
                         (float4*) (m_bvh->getNodeBuffer_dev() + nodeOfsC.x), (float4*) (m_bvh->getNodeBuffer_dev() + nodeOfsD.x),
                         (float4*) (m_bvh->getTriWoopBuffer_dev() + triOfsA.x), (float4*) (m_bvh->getTriWoopBuffer_dev() + triOfsB.x),
                         (float4*) (m_bvh->getTriWoopBuffer_dev() + triOfsC.x), (int*) m_bvh->getTriIndexBuffer_dev());


#if defined(DUMP_RAYS)
    rays.dumpRayResult();
#endif
    
    return trace_time;
}

//------------------------------------------------------------------------

/*void CudaTracer::setKernel(const String& kernelName)
{
    // Not changed => done.

    if (m_kernelName == kernelName)
        return;
    m_kernelName = kernelName;

    // Compile kernel.

    CudaModule* module = compileKernel();

    // Initialize config with default values.
    {
        KernelConfig& c         = *(KernelConfig*)module->getGlobal("g_config").getMutablePtr();
        c.bvhLayout             = BVHLayout_Max;
        c.blockWidth            = 0;
        c.blockHeight           = 0;
        c.usePersistentThreads  = 0;
    }

    // Query config.

    module->getKernel("queryConfig").launch(1, 1);
    m_kernelConfig = *(const KernelConfig*)module->getGlobal("g_config").getPtr();
}*/

//------------------------------------------------------------------------

/*
//------------------------------------------------------------------------

CudaModule* CudaTracer::compileKernel(void)
{
    m_compiler.setSourceFile(sprintf("src/rt/kernels/%s.cu", m_kernelName.getPtr()));
    m_compiler.clearDefines();
    CudaModule* module = m_compiler.compile();
    return module;
}
*/
//------------------------------------------------------------------------
