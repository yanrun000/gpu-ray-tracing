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

#include "gpu/CudaKernel.hh"

namespace FW
{

//------------------------------------------------------------------------

CudaKernel::CudaKernel(void)
:   m_gridSize          (1, 1),
    m_blockSize         (1, 1)
{
}

//------------------------------------------------------------------------

CudaKernel::~CudaKernel(void)
{
}

//------------------------------------------------------------------------

Vec2i CudaKernel::getDefaultBlockSize(void) const
{
   

    // Otherwise => guess based on GPU architecture.

    return Vec2i(32, 8);
}

//------------------------------------------------------------------------

void CudaKernel::setGrid(int numThreads, const Vec2i& blockSize)
{
    FW_ASSERT(numThreads >= 0);
    //m_blockSize = (min(blockSize) > 0) ? blockSize : getDefaultBlockSize();
    m_blockSize.x = 32; m_blockSize.y = 8;

    int maxGridWidth = 2147483647;
/*#if FW_USE_CUDA
    int tmp = CudaModule::getDeviceAttribute(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X);
    if (tmp != 0)
        maxGridWidth = tmp;
#endif*/    

    int threadsPerBlock = m_blockSize.x * m_blockSize.y;
    m_gridSize = Vec2i((numThreads + threadsPerBlock - 1) / threadsPerBlock, 1);
    while (m_gridSize.x > maxGridWidth)
    {
        m_gridSize.x = (m_gridSize.x + 1) >> 1;
        m_gridSize.y <<= 1;
    }
}

//------------------------------------------------------------------------

void CudaKernel::setGrid(const Vec2i& sizeThreads, const Vec2i& blockSize)
{
    FW_ASSERT(min(sizeThreads) >= 0);
    m_blockSize = (min(blockSize) > 0) ? blockSize : getDefaultBlockSize();
    m_gridSize = (sizeThreads + m_blockSize - 1) / m_blockSize;
}

//------------------------------------------------------------------------

}