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

#define ray_nums 2073600
#define node_nums 105925
// #define printf_ray_o
// #define printf_ray_d

// #define printf_ray_idir
// #define printf_ray_ood

#define printf_nodes_A

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

//------------------------------------------------------------------------

#if defined(printf_ray_o)//是否输出光线origx到文件

    float* ray_origx;
    ray_origx = (float*)malloc(sizeof(float)*2073600);
    ray_origx = fetch_origx((float4*) rays.getRayBuffer_dev());

    float* ray_origy;
    ray_origy = (float*)malloc(sizeof(float)*2073600);  
    ray_origy = fetch_origy((float4*) rays.getRayBuffer_dev());

    float* ray_origz;
    ray_origz = (float*)malloc(sizeof(float)*2073600);  
    ray_origz = fetch_origz((float4*) rays.getRayBuffer_dev());

    float* ray_origw;
    ray_origw = (float*)malloc(sizeof(float)*2073600);  
    ray_origw = fetch_origw((float4*) rays.getRayBuffer_dev());
   
    FILE* fp_origix = NULL;
    FILE* fp_origiy = NULL;
    FILE* fp_origiz = NULL;
    FILE* fp_origiw = NULL;
    fp_origix = fopen("origx_h.txt","w+");
    fp_origiy = fopen("origy_h.txt","w+");
    fp_origiz = fopen("origz_h.txt","w+");
    fp_origiw = fopen("tmin.txt","w+");

 
    for(int j = 0; j < 2073600; j++) 
    {
        fprintf(fp_origix, "%X\n",floatToBits(ray_origx[j]));
        fprintf(fp_origiy, "%X\n",floatToBits(ray_origy[j]));
        fprintf(fp_origiz, "%X\n",floatToBits(ray_origz[j]));
        fprintf(fp_origiw, "%X\n",floatToBits(ray_origw[j]));
    }

    fclose(fp_origix);
    fclose(fp_origiy);
    fclose(fp_origiz);
    fclose(fp_origiw);
    free(ray_origx);
    free(ray_origy);
    free(ray_origz);
    free(ray_origw);

#endif 

//------------------------------------------------------------------------

#if defined(printf_ray_d)//是否输出光线dirx到文件

    float* ray_dirx;
    ray_dirx = (float*)malloc(sizeof(float)*2073600);
    ray_dirx = fetch_dirx((float4*) rays.getRayBuffer_dev());

    float* ray_diry;
    ray_diry = (float*)malloc(sizeof(float)*2073600);  
    ray_diry = fetch_diry((float4*) rays.getRayBuffer_dev());

    float* ray_dirz;
    ray_dirz = (float*)malloc(sizeof(float)*2073600);  
    ray_dirz = fetch_dirz((float4*) rays.getRayBuffer_dev());

    float* ray_dirw;
    ray_dirw = (float*)malloc(sizeof(float)*2073600);  
    ray_dirw = fetch_dirw((float4*) rays.getRayBuffer_dev());
   
    FILE* fp_dirx = NULL;
    FILE* fp_diry = NULL;
    FILE* fp_dirz = NULL;
    FILE* fp_dirw = NULL;
    fp_dirx = fopen("dirx_h.txt","w+");
    fp_diry = fopen("diry_h.txt","w+");
    fp_dirz = fopen("dirz_h.txt","w+");
    fp_dirw = fopen("hitT.txt","w+");

 
    for(int j = 0; j < 2073600; j++) 
    {
        fprintf(fp_dirx, "%X\n",floatToBits(ray_dirx[j]));
        fprintf(fp_diry, "%X\n",floatToBits(ray_diry[j]));
        fprintf(fp_dirz, "%X\n",floatToBits(ray_dirz[j]));
        fprintf(fp_dirw, "%X\n",floatToBits(ray_dirw[j]));
    }

    fclose(fp_dirx);
    fclose(fp_diry);
    fclose(fp_dirz);
    fclose(fp_dirw);
    free(ray_dirx);
    free(ray_diry);
    free(ray_dirz);
    free(ray_dirw);

#endif 


//------------------------------------------------------------------------
#if defined(printf_ray_idir)//是否输出光线idir到文件

    float* ray_idirx;
    ray_idirx = (float*)malloc(sizeof(float)*2073600);
    ray_idirx = fetch_idirx((float4*) rays.getRayBuffer_dev());

    float* ray_idiry;
    ray_idiry = (float*)malloc(sizeof(float)*2073600);  
    ray_idiry = fetch_idiry((float4*) rays.getRayBuffer_dev());

    float* ray_idirz;
    ray_idirz = (float*)malloc(sizeof(float)*2073600);  
    ray_idirz = fetch_idirz((float4*) rays.getRayBuffer_dev());
   
    FILE* fp_idirx = NULL;
    FILE* fp_idiry = NULL;
    FILE* fp_idirz = NULL;
    fp_idirx = fopen("idirx_h.txt","w+");
    fp_idiry = fopen("idiry_h.txt","w+");
    fp_idirz = fopen("idirz_h.txt","w+");

    for(int j = 0; j < 2073600; j++) 
    {
        fprintf(fp_idirx, "%X\n",floatToBits(ray_idirx[j]));
        fprintf(fp_idiry, "%X\n",floatToBits(ray_idiry[j]));
        fprintf(fp_idirz, "%X\n",floatToBits(ray_idirz[j]));
    }

    fclose(fp_idirx);
    fclose(fp_idiry);
    fclose(fp_idirz);
    free(ray_idirx);
    free(ray_idiry);
    free(ray_idirz);

#endif 

//------------------------------------------------------------------------
#if defined(printf_ray_ood)//是否输出光线ood到文件

    float* ray_oodx;
    ray_oodx = (float*)malloc(sizeof(float)*2073600);
    ray_oodx = fetch_oodx((float4*) rays.getRayBuffer_dev());

    float* ray_oody;
    ray_oody = (float*)malloc(sizeof(float)*2073600);  
    ray_oody = fetch_oody((float4*) rays.getRayBuffer_dev());

    float* ray_oodz;
    ray_oodz = (float*)malloc(sizeof(float)*2073600);  
    ray_oodz = fetch_oodz((float4*) rays.getRayBuffer_dev());
   
    FILE* fp_oodx = NULL;
    FILE* fp_oody = NULL;
    FILE* fp_oodz = NULL;
    fp_oodx = fopen("oodx_h.txt","w+");
    fp_oody = fopen("oody_h.txt","w+");
    fp_oodz = fopen("oodz_h.txt","w+");

    for(int j = 0; j < 2073600; j++) 
    {
        fprintf(fp_oodx, "%X\n",floatToBits(ray_oodx[j]));
        fprintf(fp_oody, "%X\n",floatToBits(ray_oody[j]));
        fprintf(fp_oodz, "%X\n",floatToBits(ray_oodz[j]));
    }

    fclose(fp_oodx);
    fclose(fp_oody);
    fclose(fp_oodz);
    free(ray_oodx);
    free(ray_oody);
    free(ray_oodz);

#endif 


#if defined(printf_nodes_A)//是否输出内部节点
    float* n0xy_x;
    n0xy_x = (float*)malloc(sizeof(float)*node_nums);
    n0xy_x = fetch_n0xyx((float4*) (m_bvh->getNodeBuffer_dev() + nodeOfsA.x));

    FILE* fp_n0xy_x = NULL;
    // FILE* fp_oody = NULL;
    // FILE* fp_oodz = NULL;
    fp_n0xy_x = fopen("n0xy_x.txt","w+");
    // fp_oody = fopen("oody_h.txt","w+");
    // fp_oodz = fopen("oodz_h.txt","w+");

    // for(int j = 0; j < node_nums; j++) 
    // {
        // fprintf(fp_n0xy_x, "%X\n",floatToBits(n0xy_x[j]));
        // fprintf(fp_oody, "%X\n",floatToBits(ray_oody[j]));
        // fprintf(fp_oodz, "%X\n",floatToBits(ray_oodz[j]));
    // }

    // fclose(fp_n0xy_x);
    // fclose(fp_oody);
    // free(n0xy_x);
    // free(ray_oody);
    // free(ray_oodz);
#endif 

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
