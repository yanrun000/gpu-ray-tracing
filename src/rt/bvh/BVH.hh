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

#ifndef BVH_HH
#define BVH_HH

#include "Scene.hh"
#include "bvh/BVHNode.hh"
#include "bvh/Platform.hh"
#include "ray/RayBuffer.hh"

namespace FW
{

struct RayStats
{
    RayStats()          { clear(); }
    void clear()        { memset(this,0,sizeof(RayStats)); }
    void print() const  { if(numRays>0) printf("Ray stats: (%s) %d rays, %.1f tris/ray, %.1f nodes/ray (cost=%.2f) %.2f treelets/ray\n", platform.getName().getPtr(), numRays, 1.f*numTriangleTests/numRays, 1.f*numNodeTests/numRays, (platform.getSAHTriangleCost()*numTriangleTests/numRays + platform.getSAHNodeCost()*numNodeTests/numRays), 1.f*numTreelets/numRays ); }

    S32         numRays;
    S32         numTriangleTests;
    S32         numNodeTests;
    S32         numTreelets;
    Platform    platform;           // set by whoever sets the stats
};

class BVH
{
public:
    struct Stats
    {
        Stats()             { clear(); }
        void clear()        { memset(this, 0, sizeof(Stats)); }
        void print() const  { printf("Tree stats: [bfactor=%d] %d nodes (%d+%d), %.2f SAHCost, %.1f children/inner, %.1f tris/leaf, %d tris\n", branchingFactor,numLeafNodes+numInnerNodes, numLeafNodes,numInnerNodes, SAHCost, 1.f*numChildNodes/max(numInnerNodes,1), 1.f*numTris/max(numLeafNodes,1), numTris); }

        F32     SAHCost;
        S32     branchingFactor;
        S32     numInnerNodes;
        S32     numLeafNodes;
        S32     numChildNodes;
        S32     numTris;
    };

    struct BuildParams
    {
        Stats*      stats;
        bool        enablePrints;
        F32         splitAlpha;     // spatial split area threshold

        BuildParams(void)
        {
            stats           = NULL;
            enablePrints    = true;
            splitAlpha      = 1.0e-5f;
        }

        U32 computeHash(void) const
        {
            return hashBits(floatToBits(splitAlpha));
        }
    };

public:
                        BVH                     (Scene* scene, const Platform& platform, const BuildParams& params);
                        ~BVH                    (void)                  { if(m_root) m_root->deleteSubtree(); }

    Scene*              getScene                (void) const            { return m_scene; }
    const Platform&     getPlatform             (void) const            { return m_platform; }
    BVHNode*            getRoot                 (void) const            { return m_root; }
    void                trace                   (RayBuffer& rays, RayStats* stats = NULL) const;

    Array<S32>&         getTriIndices           (void)                  { return m_triIndices; }
    const Array<S32>&   getTriIndices           (void) const            { return m_triIndices; }

private:
    void                traceRecursive          (BVHNode* node, Ray& ray, RayResult& result, bool needClosestHit, RayStats* stats) const;

    Scene*              m_scene;
    Platform            m_platform;

    BVHNode*            m_root;
    Array<S32>          m_triIndices;
};

}

#endif //BVH_HH