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

#include <sys/stat.h>

#include "cuda/Renderer.hh"
#include "cuda/RendererKernels.hh"
#include "base/Sort.hh"
#include <iostream>///// added by yanrun 
#include <stdio.h>
#include <cuda_runtime.h>
#include "kernels/CudaTracerKernels.hh"

// #include "ray/RayBuffer.hh"

using namespace FW;

//------------------------------------------------------------------------

Renderer::Renderer(const Vec2i& frameSize)
:   //m_raygen            (3 * 640 * 480),
    m_raygen            (1 << 21),
    m_mesh              (NULL),
    m_scene             (NULL),
    m_bvh               (NULL)
{
    m_bvhCachePath = "bvhcache";
    
    m_platform = Platform("GPU");
    m_platform.setLeafPreferences(1, 8);
    m_ViewSize = frameSize;    
}

//------------------------------------------------------------------------

Renderer::~Renderer(void)
{
    setMesh(NULL);
    cudaFree(d_triMaterialColor);
    cudaFree(d_triShadedColor);    
}

//------------------------------------------------------------------------

void Renderer::setMesh(MeshBase* mesh)
{
    // Same mesh => done.

    if (mesh == m_mesh)
        return;

    // Deinit scene and BVH.

    delete m_scene;
    m_scene = NULL;
    //invalidateBVH();

    // Create scene.

    m_mesh = mesh;
    if (mesh)
        m_scene = new Scene(*mesh);
    
    if(!m_scene) return;
    
    const U8* triMaterialColor = m_scene->getTriMaterialColorBuffer().getPtr();
    const U8* triShadedColor = m_scene->getTriShadedColorBuffer().getPtr();
    
    U64 tmpSize = sizeof(U32) * m_scene->getNumTriangles();
    
    cudaMalloc(&d_triMaterialColor, tmpSize);
    cudaMalloc(&d_triShadedColor, tmpSize);
    
    cudaMemcpy(d_triMaterialColor, triMaterialColor, tmpSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_triShadedColor, triShadedColor, tmpSize, cudaMemcpyHostToDevice);
}

//------------------------------------------------------------------------

void Renderer::setParams(const Params& params)
{
    m_params = params;
    m_tracer.initKernel();
}

//------------------------------------------------------------------------

void Renderer::beginFrame(const CameraControls& camera)
{
    FW_ASSERT(m_mesh);

    // Second_Ray = false;
    // Setup BVH.
    m_tracer.setBVH(getCudaBVH());//这个实例化的结果是将getCudaBVH的值赋予m_bvh

    // Setup result image.
    const Vec2i& size = m_ViewSize;
    

    // Generate primary rays.这个模块是用来生成初始光线的。可以拿出来研究一下
    
    m_raygen.primary(m_primaryRays,
        camera.getPosition(),
        invert(Mat4f::fitToView(-1.0f, 2.0f, m_ViewSize) * camera.getWorldToClip()),
        size.x, size.y, camera.getFar());

    // printf("m_shadowStartIdx = %d", m_raygen.StartIdx);m_
    // std::cout << camera.getPosition().x << std::endl;
    // std::cout << camera.getPosition().y << std::endl;
    // std::cout << camera.getPosition().z << std::endl;
    // Secondary rays enabled => trace primary rays.

    if (m_params.rayType != RayType_Primary)
    {
        m_tracer.traceBatch(m_primaryRays);    
    }

    // Initialize state.

    float4* ray_o;
    // float4* ray_d;

    cudaMalloc((void**)&ray_o, sizeof(float4)* 100);
    // cudaMalloc((void**)&ray_d, sizeof(float4)* 100);

    float4* ray_s = (float4*) m_primaryRays.getRayBuffer_dev();

    // fetch_rays <<<1, 1, 0 >>> (ray_s);

    cudaFree(ray_o);


    m_cameraFar     = camera.getFar();
    m_newBatch      = true;
    m_batchRays     = NULL;
    m_batchStart    = 0;


}


//------------------------------------------------------------------------

CudaBVH* Renderer::getCudaBVH(void)
{
    // BVH is already valid => done.

    BVHLayout layout = m_tracer.getDesiredBVHLayout();
    
    if (!m_mesh || (m_bvh && m_bvh->getLayout() == layout))
        return m_bvh;

    // Deinit.

    delete m_bvh;
    m_bvh = NULL;

    // Setup build parameters.

    BVH::Stats stats;
    m_buildParams.stats = &stats;

    // Determine cache file name.
    
    char strbuf[512];
    
    sprintf(strbuf,"%s/%08x.dat", m_bvhCachePath.getPtr(), hashBits(
        m_scene->hash(),
        m_platform.computeHash(),
        m_buildParams.computeHash(),
        layout));
        
    String cacheFileName(strbuf);

    // Cache file exists => import.
    FileRO cachefile(cacheFileName);
    
    if(!cachefile.hasError()) 
    {
        printf("Cuda BVH Cache file reading OK!\n");
        m_bvh = new CudaBVH(cachefile);
        return m_bvh;
    }
    

    // Display status.
    printf("Building BVH...\nThis will take a while.\n");

    // Build BVH.

    BVH bvh(m_scene, m_platform, m_buildParams);
    stats.print();
    m_bvh = new CudaBVH(bvh, layout);

    // Write to cache.
    mkdir(m_bvhCachePath.getPtr(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    FileWR filewr(cacheFileName);
    m_bvh->serialize(filewr);
    
    // Display status.

    printf("Done.\n\n");
    return m_bvh;
}

//------------------------------------------------------------------------

int Renderer::getTotalNumRays(void)
{
    // Casting primary rays => no degenerates.

    if (m_params.rayType == RayType_Primary)
        return m_primaryRays.getSize();
    
    CountHitsInput in;
    
    in.numRays = m_primaryRays.getSize();
    in.rayResults = m_primaryRays.getResultBuffer_dev();
    in.raysPerThread = 32;
    
    S32 numHits = launch_countHitsKernel((in.numRays - 1) / in.raysPerThread + 1, Vec2i(CountHits_BlockWidth, CountHits_BlockHeight), in);    
    
    return numHits * m_params.numSamples;
    
}

//------------------------------------------------------------------------

bool Renderer::nextBatch(void)
{
    FW_ASSERT(m_scene);

    // Clean up the previous batch.

    if (m_batchRays)
        m_batchStart += m_batchRays->getSize();
    m_batchRays = NULL;

    // Generate new batch.

    //U32 randomSeed = (m_enableRandom) ? m_random.getU32() : 0;
    
    U32 randomSeed = 0; //random is not needed
    
    switch (m_params.rayType)
    {
    case RayType_Primary:
        if (!m_newBatch)
            return false;
        m_newBatch = false;
        m_batchRays = &m_primaryRays;
        break;

    case RayType_AO:
        if (!m_raygen.ao(m_secondaryRays, m_primaryRays, *m_scene, m_params.numSamples, m_params.aoRadius, m_newBatch, randomSeed))
            return false;
        m_batchRays = &m_secondaryRays;
        break;

    
    case RayType_Diffuse:
        if (!m_raygen.ao(m_secondaryRays, m_primaryRays, *m_scene, m_params.numSamples, m_cameraFar, m_newBatch, randomSeed))
            return false;
        m_secondaryRays.setNeedClosestHit(true);
        m_batchRays = &m_secondaryRays;
        break;

    default:
        FW_ASSERT(false);
        return false;
    }

    // Sort rays.
    if (m_params.sortSecondary && m_params.rayType != RayType_Primary)
        m_batchRays->mortonSort();    
    
    return true;
}

//------------------------------------------------------------------------

F32 Renderer::traceBatch(void)
{
    FW_ASSERT(m_batchRays);
    return m_tracer.traceBatch(*m_batchRays);
}

//------------------------------------------------------------------------

/*void Renderer::debug_updateResult(void )
{
    int io_buffer_size = 65536;
    U8* tmpPtr = new U8[65536];
    S64 tmpSize = 0;
    
    //primarySlotToID
    FILE* fchk_primarySlotToID = fopen("primarySlotToID.chk", "rb");
    if(fchk_primarySlotToID != NULL)
    {
        fread(&tmpSize, sizeof(S64), 1, fchk_primarySlotToID);
        
        h_primarySlotToID = new U8[tmpSize];
        cudaMalloc(&d_primarySlotToID, tmpSize);
        
        S64 ofs = 0; int num = 0;
        while (ofs < tmpSize)
        {
            if(tmpSize - ofs < io_buffer_size) num = tmpSize - ofs;
            else num = io_buffer_size;
            fread(tmpPtr, num, 1, fchk_primarySlotToID);
            memcpy(h_primarySlotToID + ofs, tmpPtr, num);
            ofs += num;
        }

        fclose(fchk_primarySlotToID);
        
        cudaMemcpy(d_primarySlotToID, h_primarySlotToID, tmpSize, cudaMemcpyHostToDevice);
    }else
    {
        fail("FILE primarySlotToID.chk open error!\n");
    }
    
    //primaryResults
    FILE* fchk_primaryResults = fopen("primaryResults.chk", "rb");
    if(fchk_primaryResults != NULL)
    {
        fread(&tmpSize, sizeof(S64), 1, fchk_primaryResults);
        
        h_primaryResults = new U8[tmpSize];
        cudaMalloc(&d_primaryResults, tmpSize);
        
        S64 ofs = 0; int num = 0;
        while (ofs < tmpSize)
        {
            if(tmpSize - ofs < io_buffer_size) num = tmpSize - ofs;
            else num = io_buffer_size;
            fread(tmpPtr, num, 1, fchk_primaryResults);
            memcpy(h_primaryResults + ofs, tmpPtr, num);
            ofs += num;
        }

        fclose(fchk_primaryResults);
        
        cudaMemcpy(d_primaryResults, h_primaryResults, tmpSize, cudaMemcpyHostToDevice);
    }else
    {
        fail("FILE primaryResults.chk open error!\n");
    }
    
    //batchIDToSlot
    FILE* fchk_batchIDToSlot = fopen("batchIDToSlot.chk", "rb");
    if(fchk_batchIDToSlot != NULL)
    {
        fread(&tmpSize, sizeof(S64), 1, fchk_batchIDToSlot);
        
        h_batchIDToSlot = new U8[tmpSize];
        cudaMalloc(&d_batchIDToSlot, tmpSize);
        
        S64 ofs = 0; int num = 0;
        while (ofs < tmpSize)
        {
            if(tmpSize - ofs < io_buffer_size) num = tmpSize - ofs;
            else num = io_buffer_size;
            fread(tmpPtr, num, 1, fchk_batchIDToSlot);
            memcpy(h_batchIDToSlot + ofs, tmpPtr, num);
            ofs += num;
        }

        fclose(fchk_batchIDToSlot);
        
        cudaMemcpy(d_batchIDToSlot, h_batchIDToSlot, tmpSize, cudaMemcpyHostToDevice);
    }else
    {
        fail("FILE batchIDToSlot.chk open error!\n");
    }
    
    //batchResults
    FILE* fchk_batchResults = fopen("batchResults.chk", "rb");
    if(fchk_batchResults != NULL)
    {
        fread(&tmpSize, sizeof(S64), 1, fchk_batchResults);
        
        h_batchResults = new U8[tmpSize];
        cudaMalloc(&d_batchResults, tmpSize);
        
        S64 ofs = 0; int num = 0;
        while (ofs < tmpSize)
        {
            if(tmpSize - ofs < io_buffer_size) num = tmpSize - ofs;
            else num = io_buffer_size;
            fread(tmpPtr, num, 1, fchk_batchResults);
            memcpy(h_batchResults + ofs, tmpPtr, num);
            ofs += num;
        }

        fclose(fchk_batchResults);
        
        cudaMemcpy(d_batchResults, h_batchResults, tmpSize, cudaMemcpyHostToDevice);
    }else
    {
        fail("FILE batchResults.chk open error!\n");
    }
}
*/
//------------------------------------------------------------------------

void Renderer::updateResult(U32* dev_imagePtr)
{
    FW_ASSERT(dev_imagePtr && m_batchRays);
    
    ReconstructInput in;
    
    in.numRaysPerPrimary    = (m_params.rayType == RayType_Primary) ? 1 : m_params.numSamples;
    in.firstPrimary         = m_batchStart / in.numRaysPerPrimary;
    in.numPrimary           = m_batchRays->getSize() / in.numRaysPerPrimary;
    in.isPrimary            = (m_params.rayType == RayType_Primary);
    in.isAO                 = (m_params.rayType == RayType_AO);
    in.isDiffuse            = (m_params.rayType == RayType_Diffuse);
    
    
    in.primarySlotToID      = m_primaryRays.getSlotToIDBuffer_dev();
    in.primaryResults       = m_primaryRays.getResultBuffer_dev();
    in.batchIDToSlot        = m_batchRays->getIDToSlotBuffer_dev();
    in.batchResults         = m_batchRays->getResultBuffer_dev();
    in.triMaterialColor     = d_triMaterialColor;
    in.triShadedColor       = d_triShadedColor;
    in.pixels               = dev_imagePtr;
    
    launch_reconstructKernel(in.numPrimary, in);
    
}

//------------------------------------------------------------------------
