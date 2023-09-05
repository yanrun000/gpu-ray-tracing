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

#ifndef CUDAKERNEL_HH
#define CUDAKERNEL_HH

#include "base/Math.hh"

namespace FW
{
//------------------------------------------------------------------------


//------------------------------------------------------------------------
// Automatic translation of kernel parameters:
//
// MyType*                          => mutable CUdeviceptr, valid for ONE element
// CudaKernel::Param(MyType*, N)    => mutable CUdeviceptr, valid for N elements
// Array<MyType>                    => mutable CUdeviceptr, valid for all elements
// Buffer                           => mutable CUdeviceptr, valid for the whole buffer
// Image                            => mutable CUdeviceptr, valid for all pixels
// MyType                           => MyType, passed by value (int, float, Vec4f, etc.)
//------------------------------------------------------------------------

class CudaKernel
{

public:
                        CudaKernel          (void);
                        ~CudaKernel         (void);

    Vec2i               getDefaultBlockSize (void) const;                                           // Smallest block that reaches maximal occupancy.
    void                setGrid             (int numThreads, const Vec2i& blockSize = 0);           // Generates at least numThreads.
    void                setGrid             (const Vec2i& sizeThreads, const Vec2i& blockSize = 0); // Generates at least sizeThreads in both X and Y
    
    Vec2i               getGridSize         (void) {return m_gridSize;};
    Vec2i               getBlockSize        (void) {return m_blockSize;};

private:
    Vec2i               m_gridSize;
    Vec2i               m_blockSize;
};

//------------------------------------------------------------------------
}

#endif //CUDAKERNEL_HH