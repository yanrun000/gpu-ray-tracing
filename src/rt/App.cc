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

#include "cutil_inline_runtime.h"
#include <cuda_gl_interop.h>

#include <GL/glut.h>
#include <GL/glx.h>
#include <GL/glext.h>
#define GET_PROC_ADDRESS( str ) glXGetProcAddress( (const GLubyte *)str )

#include "App.hh"
#include "3d/Mesh.hh"
#include "3d/CameraControls.hh"
#include "bvh/BVH.hh"
#include "gpu/CudaModule.hh"

#define DISPLAY_RESULT 0

//Ray type: 0 -- Primary; 1 -- AO; 2 -- Diffuse
#define RAY_TYPE 0

GLuint                      glBufferObj;
cudaGraphicsResource        *cudaResource;
FW::U32*                    dev_imagePtr = NULL;

//FW::Vec2i                   imageSize(1024, 768);
FW::Vec2i                   imageSize(1920, 1080);
//FW::Vec2i                   imageSize(640, 480);
    
PFNGLBINDBUFFERARBPROC      glBindBuffer = NULL;
PFNGLDELETEBUFFERSARBPROC   glDeleteBuffers = NULL;
PFNGLGENBUFFERSARBPROC      glGenBuffers = NULL;
PFNGLBUFFERDATAARBPROC      glBufferData = NULL;

//------------------------------------------------------------------------

void initCudaGL(void)
{
    cudaDeviceProp prop;
    int dev;
    memset( &prop, 0, sizeof( cudaDeviceProp ) );
    prop.major = 1;
    prop.minor = 0;
    cudaChooseDevice( &dev, &prop );
    cudaGLSetGLDevice( dev );
    
    char fakeParam[] = "Program";
    char *fakeargv[] = { fakeParam, NULL };
    int fakeargc = 1;
    
    if(imageSize.x == 0 || imageSize.y == 0)
        FW::fail("imageSize was not properly initialized.");

    glutInit( &fakeargc, fakeargv );
    glutInitDisplayMode( GLUT_DOUBLE | GLUT_RGBA );
    glutInitWindowSize( imageSize.x, imageSize.y );
    glutCreateWindow( "GPU Ray Tracing" );
    
    glBindBuffer    = (PFNGLBINDBUFFERARBPROC)GET_PROC_ADDRESS("glBindBuffer");
    glDeleteBuffers = (PFNGLDELETEBUFFERSARBPROC)GET_PROC_ADDRESS("glDeleteBuffers");
    glGenBuffers    = (PFNGLGENBUFFERSARBPROC)GET_PROC_ADDRESS("glGenBuffers");
    glBufferData    = (PFNGLBUFFERDATAARBPROC)GET_PROC_ADDRESS("glBufferData");
    
    glGenBuffers( 1, &glBufferObj );
    glBindBuffer( GL_PIXEL_UNPACK_BUFFER_ARB, glBufferObj );
    glBufferData( GL_PIXEL_UNPACK_BUFFER_ARB, imageSize.x * imageSize.y * 4, NULL, GL_DYNAMIC_DRAW_ARB );
    
    size_t size;
    
    cutilSafeCall(cudaGraphicsGLRegisterBuffer( &cudaResource, glBufferObj, cudaGraphicsMapFlagsNone ));
    cutilSafeCall(cudaGraphicsMapResources( 1, &cudaResource, NULL ));
    cutilSafeCall(cudaGraphicsResourceGetMappedPointer( (void**)&dev_imagePtr, &size, cudaResource) );
}

//------------------------------------------------------------------------

void glut_key_func(unsigned char key, int x, int y)
{
    switch (key) {
        case 27:
            // clean up OpenGL and CUDA
            cutilSafeCall( cudaGraphicsUnregisterResource( cudaResource ) );
            glBindBuffer( GL_PIXEL_UNPACK_BUFFER_ARB, 0 );
            glDeleteBuffers( 1, &glBufferObj );
            exit(0);
    }
}

//------------------------------------------------------------------------

void glut_draw_func(void )
{
    glDrawPixels( imageSize.x, imageSize.y, GL_RGBA, GL_UNSIGNED_BYTE, 0 );
    glutSwapBuffers();
}

//------------------------------------------------------------------------

void displayResult(void)
{
    cutilSafeCall( cudaGraphicsUnmapResources( 1, &cudaResource, NULL ) );

    // set up GLUT and kick off main loop
    glutKeyboardFunc( glut_key_func );
    glutDisplayFunc( glut_draw_func );
    glutMainLoop();
}

//------------------------------------------------------------------------


void FW::runBenchmark(
    const String&           meshFile,
    const Array<String>&    cameras,
    F32                     sbvhAlpha,
    F32                     aoRadius,
    int                     numSamples,
    bool                    sortSecondary,
    int                     warmupRepeats,
    int                     measureRepeats)
{
    //TODO:CUDA & GL Initialize
    CudaModule::staticInit();
    
    //int numRayTypes = Renderer::RayType_Max;
    
    // Setup renderer.
    Renderer::Params params;
    params.aoRadius = aoRadius;
    params.numSamples = numSamples;
    //params.sortSecondary = sortSecondary;
    params.sortSecondary = false;

    BVH::BuildParams buildParams;
    buildParams.splitAlpha = sbvhAlpha;

    Renderer renderer(imageSize);
    renderer.setBuildParams(buildParams);
    renderer.setMesh(importMesh(meshFile));
    
#ifdef DISPLAY_RESULT
    initCudaGL();
#endif
    
    //Cameras are moved out of the loop, because there is always one camera.
    CameraControls camera;
    camera.decodeSignature(cameras[0]);//～～～～～～～～～～～这里设置了camera
    //Cameras are moved out of the loop, because there is always one camera.
    
    int rayType = RAY_TYPE;//设置默认的光线类型
    
    S64 totalRays = 0;
    F32 totalLaunchTime = 0.0f;
        
    params.rayType = (Renderer::RayType)rayType;
    renderer.setParams(params);
        
    renderer.beginFrame(camera);
        
    //totalRays += (S64)renderer.getTotalNumRays() * measureRepeats;
    totalRays += (S64)renderer.getTotalNumRays();
        
    printf("totalRays = %d\n", totalRays);
    
    float totalTracingTime = 0.0f;
            
    while(renderer.nextBatch())
    {
        totalTracingTime += renderer.traceBatch();//这个意味着trace的时间在增加
#ifdef DISPLAY_RESULT
        renderer.updateResult(dev_imagePtr);
#endif
    }    
    
    printf("Results = %.2f M Rays/s\n", (float) totalRays /((1000.0f) * totalTracingTime));

#ifdef DISPLAY_RESULT
    displayResult();
#endif

}

//------------------------------------------------------------------------
