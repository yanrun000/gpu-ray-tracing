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

#include "ray/RayBuffer.hh"
#include "ray/RayBufferKernels.hh"
#include "base/Math.hh"
#include "base/Sort.hh"
//#include "base/Random.hh"
//#include "gpu/CudaCompiler.hh"

#include <cuda_runtime.h>

#include <stdio.h>

namespace FW
{

RayBuffer::~RayBuffer(void )
{
    if(m_size != 0)
    {
        cudaFree(m_rays_dev);
        cudaFree(m_IDToSlot_dev);
        cudaFree(m_slotToID_dev);
        cudaFree(m_results_dev);
    }
}
    
void RayBuffer::resize(S32 n)
{
    FW_ASSERT(n >= 0);
    
    if (n < m_size)
    {
        m_size = n;
        return;
    }
    
    if(m_size !=0)
    {
        cudaFree(m_rays_dev);
        cudaFree(m_IDToSlot_dev);
        cudaFree(m_slotToID_dev);
        cudaFree(m_results_dev);
    }

    m_size = n;
    m_rays.resize(n * sizeof(Ray));
    m_results.resize(n * sizeof(RayResult));
    m_IDToSlot.resize(n * sizeof(S32));
    m_slotToID.resize(n * sizeof(S32));
    
    cudaMalloc(&m_rays_dev, n * sizeof(Ray));
    cudaMalloc(&m_IDToSlot_dev, n * sizeof(S32));
    cudaMalloc(&m_slotToID_dev, n * sizeof(S32));
    cudaMalloc(&m_results_dev, n * sizeof(RayResult));
}

void RayBuffer::copyRays(Ray* dumpRays)
{
    cudaMemcpy(dumpRays, m_rays_dev, m_size * sizeof(Ray), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
}

void RayBuffer::dumpRayBuffer()
{
    static int dr_cnt = 0;   
    
    /*if(dr_cnt == 0)
    {
        dr_cnt ++;
        return;
    }*/
    
    char dumpfilename[100];
    
    sprintf(dumpfilename, "AORay%02d.dump", dr_cnt);
    
    FILE* fraydump;
    
    fraydump = fopen(dumpfilename, "w");
    if(fraydump == NULL)
    {
        printf("Can't open %s!\n", dumpfilename);
        exit(-1);
    }
    
    fwrite(&m_size, sizeof(int), 1, fraydump);

    Ray* raybuffer = new Ray[m_size];
    cudaMemcpy(raybuffer, m_rays_dev, m_size * sizeof(Ray), cudaMemcpyDeviceToHost);
    
    cudaDeviceSynchronize();
    
    int itemleft = m_size;
    int wptr = 0;
    
    while(itemleft >0)
    {
        if(itemleft > 128)
        {
            fwrite((raybuffer + wptr), sizeof(Ray), 128, fraydump);
            itemleft -= 128;
            wptr += 128;
        }else
        {
            fwrite((raybuffer + wptr), sizeof(Ray), itemleft, fraydump);
            itemleft = 0;
        }
    }
    
    fclose(fraydump);
    
    /*Ray* raybuffer = new Ray[m_size];
    cudaMemcpy(raybuffer, m_rays_dev, m_size * sizeof(Ray), cudaMemcpyDeviceToHost);
    
    cudaDeviceSynchronize();
    
    FILE* fraydumptxt;
    fraydumptxt = fopen("AORay.txt", "w");
    if(fraydumptxt == NULL)
    {
        printf("Can't open AORay.txt!\n");
        return;
    }
    fprintf(fraydumptxt, "%d\n", m_size);
    for(int i = 0; i < m_size; i ++)
    {
        fprintf(fraydumptxt, "%d: origin:%f %f %f; tmin:%f; dir:%f %f %f; tmax:%f\n", 
                i, raybuffer[i].origin.x, raybuffer[i].origin.y, raybuffer[i].origin.z, raybuffer[i].tmin,
                raybuffer[i].direction.x, raybuffer[i].direction.y, raybuffer[i].direction.z, raybuffer[i].tmax);
    }
    fclose(fraydumptxt);*/
    
    dr_cnt ++;
}

void RayBuffer::dumpRayResult()
{
    FILE* fraydump;
    static int rr_cnt = 0;
    
    printf("RaySize = %d\n", m_size);
    
    
    /*if(rr_cnt == 0)
    {
        rr_cnt ++;
        return;
    }*/
    
    char dumpfilename[100];
    
    sprintf(dumpfilename, "RayResult%02d.dump", rr_cnt);
    
    fraydump = fopen(dumpfilename, "w");
    if(fraydump == NULL)
    {
        printf("Can't open %s!\n", dumpfilename);
        exit(-1);
    }
    
    //fwrite(&m_size, sizeof(int), 1, fraydump);
    fprintf(fraydump, "%d\n", m_size);
    
    RayResult* rayresult = new RayResult[m_size];
    cudaMemcpy(rayresult, m_results_dev, m_size * sizeof(RayResult), cudaMemcpyDeviceToHost);
    
    cudaDeviceSynchronize();
    
    for(int i = 0; i < m_size; i ++)
    {
        fprintf(fraydump, "%d %f\n", rayresult[i].id, rayresult[i].t);
    }

    /*
    int itemleft = m_size;
    int wptr = 0;
    
    while(itemleft >0)
    {
        if(itemleft > 128)
        {
            fwrite((rayresult + wptr), sizeof(RayResult), 128, fraydump);
            itemleft -= 128;
            wptr += 128;
        }else
        {
            fwrite((rayresult + wptr), sizeof(RayResult), itemleft, fraydump);
            itemleft = 0;
        }
    }*/
    
    fclose(fraydump);
    
    //exit(-1);
    
    rr_cnt ++;
}



void RayBuffer::setRay(S32 slot, const Ray& ray, S32 id)
{
    FW_ASSERT(slot >= 0 && slot < m_size);
    FW_ASSERT(id >= 0 && id < m_size);

    ((Ray*)m_rays.getMutablePtr())[slot] = ray;
    ((S32*)m_IDToSlot.getMutablePtr())[id] = slot;
    ((S32*)m_slotToID.getMutablePtr())[slot] = id;
}

static bool compareMortonKey(void* data, int idxA, int idxB)
{
    const MortonKey& a = ((const MortonKey*)data)[idxA];
    const MortonKey& b = ((const MortonKey*)data)[idxB];
    if (a.hash[5] != b.hash[5]) return (a.hash[5] < b.hash[5]);
    if (a.hash[4] != b.hash[4]) return (a.hash[4] < b.hash[4]);
    if (a.hash[3] != b.hash[3]) return (a.hash[3] < b.hash[3]);
    if (a.hash[2] != b.hash[2]) return (a.hash[2] < b.hash[2]);
    if (a.hash[1] != b.hash[1]) return (a.hash[1] < b.hash[1]);
    if (a.hash[0] != b.hash[0]) return (a.hash[0] < b.hash[0]);
    return false;
}

void RayBuffer::getRayAABB(Vec3f& aabbLo, Vec3f& aabbHi)
{
    aabbLo = rayAABBLo;
    aabbHi = rayAABBHi;
}

void RayBuffer::mortonSort()
{   
    FindAABBOutput aabb_out;
    
    {
        FindAABBInput in;
        in.numRays          = getSize();
        in.inRays           = getRayBuffer_dev();
        in.raysPerThread    = 32;
        aabb_out.aabbLo     = Vec3f(+FW_F32_MAX);
        aabb_out.aabbHi     = Vec3f(-FW_F32_MAX);
        
        launch_findAABBKernel((in.numRays - 1) / in.raysPerThread + 1,
            Vec2i(FindAABB_BlockWidth, FindAABB_BlockHeight), in, aabb_out);
    }
    
    MortonKey* keyBuffer;
    MortonKey* keyBuffer_dev;
    
    keyBuffer = (MortonKey*) malloc(getSize() * sizeof(MortonKey));
    cudaMalloc(&keyBuffer_dev, getSize() * sizeof(MortonKey));
    
    // Generate keys.
    {
        GenMortonKeysInput in;
        in.numRays  = getSize();
        in.aabbLo   = aabb_out.aabbLo;
        in.aabbHi   = aabb_out.aabbHi;
        rayAABBLo   = aabb_out.aabbLo;
        rayAABBHi   = aabb_out.aabbHi;
        in.inRays   = getRayBuffer_dev();
        in.outKeys  = keyBuffer_dev;
        
        launch_genMortonKeysKernel(getSize(), Vec2i(32, 4), in);
        
        cudaMemcpy(keyBuffer, keyBuffer_dev, getSize() * sizeof(MortonKey), cudaMemcpyDeviceToHost);
    }
    
    // Sort keys.

    sort(keyBuffer, getSize(), compareMortonKey, sortDefaultSwap<MortonKey>, true);
    cudaMemcpy(keyBuffer_dev, keyBuffer, getSize() * sizeof(MortonKey), cudaMemcpyHostToDevice);
     
    
    //Allocate temporary buffers
    Ray* oldRay_dev;
    S32* oldSlotToID_dev;
    
    cudaMalloc(&oldRay_dev, getSize() * sizeof(Ray));
    cudaMalloc(&oldSlotToID_dev, getSize() * sizeof(S32));
    
    cudaMemcpy(oldRay_dev, m_rays_dev, getSize() * sizeof(Ray), cudaMemcpyDeviceToDevice);
    cudaMemcpy(oldSlotToID_dev, m_slotToID_dev, getSize() * sizeof(S32), cudaMemcpyDeviceToDevice);
    
    // Reorder rays.
    {
        ReorderRaysInput in;
        in.numRays      = getSize();
        in.inKeys       = keyBuffer_dev;
        in.inRays       = oldRay_dev;
        in.inSlotToID   = oldSlotToID_dev;
        in.outRays      = getRayBuffer_dev();
        in.outIDToSlot  = getIDToSlotBuffer_dev();
        in.outSlotToID  = getSlotToIDBuffer_dev();
        
        launch_reorderRaysKernel(getSize(), in);
    }
    free(keyBuffer);
}


//-------------------------------------------------------------------

/*void RayBuffer::randomSort(U32 randomSeed)
{
    // Reorder rays.

    Ray* rays = (Ray*)getRayBuffer().getMutablePtr();
    S32* idToSlot = (S32*)getIDToSlotBuffer().getMutablePtr();
    S32* slotToID = (S32*)getSlotToIDBuffer().getMutablePtr();
    Random random(randomSeed);

    for (int slot = 0; slot < m_size; slot++)
    {
        S32 slot2 = random.getS32(m_size - slot) + slot;

        S32 id  = slotToID[slot];
        S32 id2 = slotToID[slot2];

        swap(rays[slot],    rays[slot2]);
        swap(slotToID[slot],slotToID[slot2]);
        swap(idToSlot[id],  idToSlot[id2]);
    }
}*/

//-------------------------------------------------------------------

}
