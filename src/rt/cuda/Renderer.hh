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

#ifndef RENDERER_HH
#define RENDERER_HH

#include "base/Defs.hh"
#include "3d/Mesh.hh"
#include "bvh/Platform.hh"
#include "bvh/BVH.hh"
#include "cuda/CudaBVH.hh"
#include "Scene.hh"
#include "3d/CameraControls.hh"
#include "io/File.hh"
#include "gui/Image.hh"
#include "cuda/CudaTracer.hh"
//#include "base/Random.hpp"
#include "ray/RayGen.hh"
#include "Util.hh"


namespace FW
{
//------------------------------------------------------------------------


class Renderer
{
public:
    enum RayType
    {
        RayType_Primary = 0,
        RayType_AO,
        RayType_Diffuse,
        RayType_Max
    };

    struct Params
    {
        RayType         rayType;
        F32             aoRadius;
        S32             numSamples;
        bool            sortSecondary;

        Params(void)
        {
            rayType         = RayType_Primary;
            aoRadius        = 1.0f;
            numSamples      = 32;
            sortSecondary   = false;
        }
    };

public:
                        Renderer            (const Vec2i&);
                        ~Renderer           (void);

    void                setMesh             (MeshBase* mesh);
    void                setBuildParams      (const BVH::BuildParams& params) { invalidateBVH(); m_buildParams = params; }
    void                invalidateBVH       (void)                  { delete m_bvh; m_bvh = NULL; }

    void                setParams           (const Params& params);
    //void                setMessageWindow    (Window* window)        { m_window = window; m_compiler.setMessageWindow(window); m_tracer.setMessageWindow(window); }
    //void                setEnableRandom     (bool enable)           { m_enableRandom = enable; }

    CudaTracer&         getCudaTracer       (void)                  { return m_tracer; }
    Scene*              getScene            (void) const            { return m_scene; }
    CudaBVH*            getCudaBVH          (void);

    //F32                 renderFrame         (GLContext* gl, const CameraControls& camera); // returns total launch time

    void                beginFrame          (const CameraControls& camera);
    bool                nextBatch           (void);
    F32                 traceBatch          (void); // returns launch time
    void                updateResult        (U32*); // for current batch

    int                 getTotalNumRays     (void); // for selected ray type, excluding degenerates
    
    void                hashRays            (void);
    bool                second_fetch;  

private:
                        //Renderer            (const Renderer&); // forbidden
    Renderer&           operator=           (const Renderer&); // forbidden
    //void                debug_updateResult  (void);

private:
    //CudaCompiler        m_compiler;
    String              m_bvhCachePath;//
    Platform            m_platform;//
    BVH::BuildParams    m_buildParams;
    RayGen              m_raygen;
    CudaTracer          m_tracer;
    //Random              m_random;

    Params              m_params;
    //Window*             m_window;
    //bool                m_enableRandom;

    MeshBase*           m_mesh;
    Scene*              m_scene;
    CudaBVH*            m_bvh;

    Vec2i               m_ViewSize;
    F32                 m_cameraFar;
    RayBuffer           m_primaryRays;
    RayBuffer           m_secondaryRays;

    bool                m_newBatch;
    RayBuffer*          m_batchRays;
    S32                 m_batchStart;
    
    //Moved from scene directly to scene
    U32*                d_triMaterialColor;
    U32*                d_triShadedColor;  
  
};

//------------------------------------------------------------------------
}

#endif //RENDERER_HH
