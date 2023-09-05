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

#include "io/MeshWavefrontIO.hh"
#include "io/File.hh"
#include "base/Hash.hh"

using namespace FW;


//------------------------------------------------------------------------

#define WAVEFRONT_DEBUG 0   // 1 = fail on error, 0 = ignore errors

//------------------------------------------------------------------------

namespace FW
{

struct Material : public MeshBase::Material
{
    S32                     submesh;
};

struct TextureSpec
{
    F32                     base;
    F32                     gain;
};

struct ImportState
{
    Mesh<VertexPNT>*        mesh;

    Array<Vec3f>            positions;
    Array<Vec2f>            texCoords;
    Array<Vec3f>            normals;

    Hash<Vec3i, S32>        vertexHash;
    Hash<String, Material>  materialHash;

    Array<S32>              vertexTmp;
    Array<Vec3i>            indexTmp;
};

template <class T> bool equals  (const VertexPNT& a, const VertexPNT& b);
template <class T> U32 hash     (const VertexPNT& value);

static bool     parseFloats     (const char*& ptr, F32* values, int num);
static bool     parseTexture    (const char*& ptr, TextureSpec& value, const String& dirName);

static void     loadMtl         (ImportState& s, BufferedInputStream& mtlIn, const String& dirName);
static void     loadObj         (ImportState& s, BufferedInputStream& objIn, const String& dirName);

}

//------------------------------------------------------------------------

template <class T> bool FW::equals(const VertexPNT& a, const VertexPNT& b)
{
    return (equals<Vec3f>(a.p, b.p) && equals<Vec3f>(a.n, b.n) && equals<Vec2f>(a.t, b.t));
}

//------------------------------------------------------------------------

template <class T> U32 FW::hash(const VertexPNT& value)
{
    return hashBits(hash<Vec3f>(value.p), hash<Vec3f>(value.n), hash<Vec2f>(value.t));
}

//------------------------------------------------------------------------

bool FW::parseFloats(const char*& ptr, F32* values, int num)
{
    FW_ASSERT(values);
    const char* tmp = ptr;
    for (int i = 0; i < num; i++)
    {
        if (i)
            parseSpace(tmp);
        if (!parseFloat(tmp, values[i]))
            return false;
    }
    ptr = tmp;
    return true;
}

//------------------------------------------------------------------------

void FW::loadMtl(ImportState& s, BufferedInputStream& mtlIn, const String& dirName)
{
    Material* mat = NULL;
    for (int lineNum = 1;; lineNum++)
    {
        const char* line = mtlIn.readLine(true, true);
        if (!line)
            break;
        
        const char* ptr = line;
        parseSpace(ptr);
        bool valid = false;

        if (!*ptr || parseLiteral(ptr, "#"))
        {
            valid = true;
        }
        else if (parseLiteral(ptr, "newmtl ") && parseSpace(ptr) && *ptr) // material name
        {
            if (!s.materialHash.contains(ptr))
            {
                mat = &s.materialHash.add(ptr);
                mat->submesh = -1;
            }
            valid = true;
        }
        else if (!mat)
        {
#if WAVEFRONT_DEBUG
            setError("No current material in Wavefront MTL: '%s'!", line);
#endif
        }
        else if (parseLiteral(ptr, "Ka ") && parseSpace(ptr)) // ambient color
        {
            F32 tmp[3];
            if (parseLiteral(ptr, "spectral ") || parseLiteral(ptr, "xyz "))
                valid = true;
            else if (parseFloats(ptr, tmp, 3) && parseSpace(ptr) && !*ptr)
                valid = true;
        }
        else if (parseLiteral(ptr, "Kd ") && parseSpace(ptr)) // diffuse color
        {
            if (parseLiteral(ptr, "spectral ") || parseLiteral(ptr, "xyz "))
                valid = true;
            else if (parseFloats(ptr, mat->diffuse.getPtr(), 3) && parseSpace(ptr) && !*ptr)
                valid = true;
        }
        else if (parseLiteral(ptr, "Ks ") && parseSpace(ptr)) // specular color
        {
            if (parseLiteral(ptr, "spectral ") || parseLiteral(ptr, "xyz "))
                valid = true;
            else if (parseFloats(ptr, mat->specular.getPtr(), 3) && parseSpace(ptr) && !*ptr)
                valid = true;
        }
        else if (parseLiteral(ptr, "d ") && parseSpace(ptr)) // alpha
        {
            if (parseFloat(ptr, mat->diffuse.w) && parseSpace(ptr) && !*ptr)
                valid = true;
        }
        else if (parseLiteral(ptr, "Ns ") && parseSpace(ptr)) // glossiness
        {
            if (parseFloat(ptr, mat->glossiness) && parseSpace(ptr) && !*ptr)
                valid = true;
            if (mat->glossiness <= 0.0f)
            {
                mat->glossiness = 1.0f;
                mat->specular.setZero();
            }
        }
        else if (parseLiteral(ptr, "map_Kd ")) // diffuse texture
        {
            fail("parseTexture shouldn't be called");
            /*TextureSpec tex;
            valid = parseTexture(ptr, tex, dirName);
            mat->textures[MeshBase::TextureType_Diffuse] = tex.texture;*/
        }
        else if (parseLiteral(ptr, "map_d ") || parseLiteral(ptr, "map_D ") || parseLiteral(ptr, "map_opacity ")) // alpha texture
        {
            fail("parseTexture shouldn't be called");
            /*
            TextureSpec tex;
            valid = parseTexture(ptr, tex, dirName);
            mat->textures[MeshBase::TextureType_Alpha] = tex.texture;*/
        }
        else if (parseLiteral(ptr, "disp ")) // displacement map
        {
            fail("parseTexture shouldn't be called");
            /*
            TextureSpec tex;
            valid = parseTexture(ptr, tex, dirName);
            mat->displacementCoef = tex.gain;
            mat->displacementBias = tex.base * tex.gain;
            mat->textures[MeshBase::TextureType_Displacement] = tex.texture;*/
        }
        else if (parseLiteral(ptr, "bump ") || parseLiteral(ptr, "map_bump ") || parseLiteral(ptr, "map_Bump ")) // bump map
        {
            fail("parseTexture shouldn't be called");
            /*
            TextureSpec tex;
            valid = parseTexture(ptr, tex, dirName);
            mat->textures[MeshBase::TextureType_Normal] = tex.texture;*/
        }
        else if (parseLiteral(ptr, "refl ")) // environment map
        {
            fail("parseTexture shouldn't be called");
            /*
            TextureSpec tex;
            valid = parseTexture(ptr, tex, dirName);
            mat->textures[MeshBase::TextureType_Environment] = tex.texture;*/
        }
        else if (
            parseLiteral(ptr, "vp ") ||             // parameter space vertex
            parseLiteral(ptr, "Kf ") ||             // transmission color
            parseLiteral(ptr, "illum ") ||          // illumination model
            parseLiteral(ptr, "d -halo ") ||        // orientation-dependent alpha
            parseLiteral(ptr, "sharpness ") ||      // reflection sharpness
            parseLiteral(ptr, "Ni ") ||             // index of refraction
            parseLiteral(ptr, "map_Ks ") ||         // specular texture
            parseLiteral(ptr, "map_kS ") ||         // ???
            parseLiteral(ptr, "map_kA ") ||         // ???
            parseLiteral(ptr, "map_Ns ") ||         // glossiness texture
            parseLiteral(ptr, "map_aat ") ||        // texture antialiasing
            parseLiteral(ptr, "decal ") ||          // blended texture
            parseLiteral(ptr, "Km ") ||             // ???
            parseLiteral(ptr, "Tr ") ||             // ???
            parseLiteral(ptr, "Tf ") ||             // ???
            parseLiteral(ptr, "Ke ") ||             // ???
            parseLiteral(ptr, "pointgroup ") ||     // ???
            parseLiteral(ptr, "pointdensity ") ||   // ???
            parseLiteral(ptr, "smooth") ||          // ???
            parseLiteral(ptr, "R "))                // ???
        {
            valid = true;
        }

#if WAVEFRONT_DEBUG
        if (!valid)
            fail("Invalid line %d in Wavefront MTL: '%s'!", lineNum, line);
#endif
    }
}

//------------------------------------------------------------------------

void FW::loadObj(ImportState& s, BufferedInputStream& objIn, const String& dirName)
{
    int submesh = -1;
    int defaultSubmesh = -1;

    for (int lineNum = 1;; lineNum++)
    {
        const char* line = objIn.readLine(true, true);
        if (!line)
            break;

        const char* ptr = line;
        parseSpace(ptr);
        bool valid = false;

        if (!*ptr || parseLiteral(ptr, "#"))
        {
            valid = true;
        }
        else if (parseLiteral(ptr, "v ") && parseSpace(ptr)) // position vertex
        {
            Vec3f v;
            if (parseFloats(ptr, v.getPtr(), 3) && parseSpace(ptr) && !*ptr)
            {
                s.positions.add(v);
                valid = true;
            }
        }
        else if (parseLiteral(ptr, "vt ") && parseSpace(ptr)) // texture vertex
        {
            Vec2f v;
            if (parseFloats(ptr, v.getPtr(), 2) && parseSpace(ptr))
            {
                F32 dummy;
                while (parseFloat(ptr, dummy) && parseSpace(ptr));

                if (!*ptr)
                {
                    s.texCoords.add(Vec2f(v.x, 1.0f - v.y));
                    valid = true;
                }
            }
        }
        else if (parseLiteral(ptr, "vn ") && parseSpace(ptr)) // normal vertex
        {
            Vec3f v;
            if (parseFloats(ptr, v.getPtr(), 3) && parseSpace(ptr) && !*ptr)
            {
                s.normals.add(v);
                valid = true;
            }
        }
        else if (parseLiteral(ptr, "f ") && parseSpace(ptr)) // face
        {
            s.vertexTmp.clear();
            while (*ptr)
            {
                Vec3i ptn;
                if (!parseInt(ptr, ptn.x))
                    break;
                for (int i = 1; i < 4 && parseLiteral(ptr, "/"); i++)
                {
                    S32 tmp = 0;
                    parseInt(ptr, tmp);
                    if (i < 3)
                        ptn[i] = tmp;
                }
                parseSpace(ptr);

                Vec3i size(s.positions.getSize(), s.texCoords.getSize(), s.normals.getSize());
                for (int i = 0; i < 3; i++)
                {
                    if (ptn[i] < 0)
                        ptn[i] += size[i];
                    else
                        ptn[i]--;

                    if (ptn[i] < 0 || ptn[i] >= size[i])
                        ptn[i] = -1;
                }

                S32* idx = s.vertexHash.search(ptn);
                if (idx)
                    s.vertexTmp.add(*idx);
                else
                {
                    s.vertexTmp.add(s.vertexHash.add(ptn, s.mesh->numVertices()));
                    VertexPNT& v = s.mesh->addVertex();
                v.p = (ptn.x == -1) ? Vec3f(0.0f) : s.positions[ptn.x];
                v.t = (ptn.y == -1) ? Vec2f(0.0f) : s.texCoords[ptn.y];
                v.n = (ptn.z == -1) ? Vec3f(0.0f) : s.normals[ptn.z];
                }
            }
            if (!*ptr)
            {
                if (submesh == -1)
                {
                    if (defaultSubmesh == -1)
                        defaultSubmesh = s.mesh->addSubmesh();
                    submesh = defaultSubmesh;
                }
                for (int i = 2; i < s.vertexTmp.getSize(); i++)
                    s.indexTmp.add(Vec3i(s.vertexTmp[0], s.vertexTmp[i - 1], s.vertexTmp[i]));
                valid = true;
            }
        }
        else if (parseLiteral(ptr, "usemtl ") && parseSpace(ptr)) // material name
        {
            Material* mat = s.materialHash.search(ptr);
            if (submesh != -1)
            {
                s.mesh->mutableIndices(submesh).add(s.indexTmp);
                s.indexTmp.clear();
            submesh = -1;
            }
            if (mat)
            {
                if (mat->submesh == -1)
                {
                    mat->submesh = s.mesh->addSubmesh();
                    s.mesh->material(mat->submesh) = *mat;
                }
                submesh = mat->submesh;
                s.indexTmp.clear();
            }
            valid = true;
        }
        else if (parseLiteral(ptr, "mtllib ") && parseSpace(ptr) && *ptr) // material library
        {
            if (dirName.getLength())
            {
                String fileName = dirName + "/" + ptr;
                FileRO file(fileName);
                BufferedInputStream mtlIn(file);
                loadMtl(s, mtlIn, fileName.getDirName());

#if (!WAVEFRONT_DEBUG)
                //clearError();
#endif
            }
            valid = true;
        }
        else if (
            parseLiteral(ptr, "vp ") ||         // parameter space vertex
            parseLiteral(ptr, "deg ") ||        // degree
            parseLiteral(ptr, "bmat ") ||       // basis matrix
            parseLiteral(ptr, "step ") ||       // step size
            parseLiteral(ptr, "cstype ") ||     // curve/surface type
            parseLiteral(ptr, "p ") ||          // point
            parseLiteral(ptr, "l ") ||          // line
            parseLiteral(ptr, "curv ") ||       // curve
            parseLiteral(ptr, "curv2 ") ||      // 2d curve
            parseLiteral(ptr, "surf ") ||       // surface
            parseLiteral(ptr, "parm ") ||       // curve/surface parameters
            parseLiteral(ptr, "trim ") ||       // curve/surface outer trimming loop
            parseLiteral(ptr, "hole ") ||       // curve/surface inner trimming loop
            parseLiteral(ptr, "scrv ") ||       // curve/surface special curve
            parseLiteral(ptr, "sp ") ||         // curve/surface special point
            parseLiteral(ptr, "end ") ||        // curve/surface end statement
            parseLiteral(ptr, "con ") ||        // surface connect
            parseLiteral(ptr, "g ") ||          // group name
            parseLiteral(ptr, "s ") ||          // smoothing group
            parseLiteral(ptr, "mg ") ||         // merging group
            parseLiteral(ptr, "o ") ||          // object name
            parseLiteral(ptr, "bevel ") ||      // bevel interpolation
            parseLiteral(ptr, "c_interp ") ||   // color interpolation
            parseLiteral(ptr, "d_interp ") ||   // dissolve interpolation
            parseLiteral(ptr, "lod ") ||        // level of detail
            parseLiteral(ptr, "shadow_obj ") || // shadow casting
            parseLiteral(ptr, "trace_obj ") ||  // ray tracing
            parseLiteral(ptr, "ctech ") ||      // curve approximation technique
            parseLiteral(ptr, "stech ") ||      // surface approximation technique
            parseLiteral(ptr, "g"))             // ???
        {
            valid = true;
        }

#if WAVEFRONT_DEBUG
        if (!valid)
            setError("Invalid line %d in Wavefront OBJ: '%s'!", lineNum, line);
#endif
    }

    // Flush remaining indices.

    if (submesh != -1)
        s.mesh->mutableIndices(submesh).add(s.indexTmp);
}

//------------------------------------------------------------------------

Mesh<VertexPNT>* FW::importWavefrontMesh(BufferedInputStream& stream, const String& fileName)
{
    int vertexCapacity = 4 << 10;
    int indexCapacity = 4 << 10;

    ImportState s;
    s.mesh = new Mesh<VertexPNT>;
    s.mesh->resizeVertices(vertexCapacity);
    s.mesh->clearVertices();
    s.positions.setCapacity(vertexCapacity);
    s.texCoords.setCapacity(vertexCapacity);
    s.normals.setCapacity(vertexCapacity);
    s.vertexHash.setCapacity(vertexCapacity);
    s.indexTmp.setCapacity(indexCapacity);

    loadObj(s, stream, fileName.getDirName());
    s.mesh->compact();
    return s.mesh;
}


