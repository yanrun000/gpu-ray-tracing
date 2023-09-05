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
#include <stdio.h>

#include "App.hh"
#include "base/Main.hh"
#include "base/String.hh"

//using namespace FW;

//------------------------------------------------------------------------

int         FW::argc            = 0;
char**      FW::argv            = NULL;
int         FW::exitCode        = 0;

//------------------------------------------------------------------------

static const char* const s_commandHelpText =
    "\n"
    "Usage: gpu-rt [options]\n"  
    "Common options:\n"
    "\n"
    "   --log=<file.log>        Log all output to file.\n"
    "   --size=<w>x<h>          Frame size. Default is \"1024x768\".\n"
    "\n"
    "Options for \"rt benchmark\":\n"
    "\n"
    "   --mesh=<file.obj>       Mesh to benchmark.\n"
    "   --camera=\"<sig>\"        Camera signature. Can specify multiple times.\n"
    "   --sbvh-alpha=<value>    SBVH alpha parameter. Default is \"1.0e-5\".\n"
    "   --ao-radius=<value>     AO ray length. Default is \"5\".\n"
    "   --samples=<value>       Secondary rays per pixel. Default is \"32\".\n"
    "   --sort=<1/0>            Sort secondary rays. Default is \"1\".\n"
    "   --warmup-repeats=<num>  Launches prior to measurement. Default is \"2\".\n"
    "   --measure-repeats=<num> Launches to measure per batch. Default is \"10\".\n"
    "\n"
;
//------------------------------------------------------------------------

void FW::init(void)
{
    // Parse mode.

    bool showHelp           = false;

    if (argc < 1)
    {
        printf("Specify \"--help\" for a list of command-line options.\n\n");
    }
    
    // Parse options.
    String          meshFile;
    Array<String>   cameras;
    F32             sbvhAlpha       = 1.0e-5f;
    F32             aoRadius        = 5.0f;
    int             numSamples      = 8;
    bool            sortRays        = true;
    int             warmupRepeats   = 2;
    int             measureRepeats  = 10;

    for (int i = 1; i < argc; i++)
    {
        const char* ptr = argv[i];

        if ((parseLiteral(ptr, "--help") || parseLiteral(ptr, "-h")) && !*ptr)
        {
            showHelp = true;
        }        
        else if (parseLiteral(ptr, "--mesh="))
        {
            if (!*ptr)
                fail("Invalid mesh file '%s'!", argv[i]);
            meshFile = ptr;
        }
        else if (parseLiteral(ptr, "--camera="))
        {
            if (!*ptr)
                fail("Invalid camera signature '%s'!", argv[i]);
            cameras.add(ptr);
        }
        else if (parseLiteral(ptr, "--sbvh-alpha="))
        {
            if (!parseFloat(ptr, sbvhAlpha) || *ptr || sbvhAlpha < 0.0f || sbvhAlpha > 1.0f)
                fail("Invalid SBVH alpha '%s'!", argv[i]);
        }
        else if (parseLiteral(ptr, "--ao-radius="))
        {
            if (!parseFloat(ptr, aoRadius) || *ptr || aoRadius < 0.0f)
                fail("Invalid AO radius '%s'!", argv[i]);
        }
        else if (parseLiteral(ptr, "--samples="))
        {
            if (!parseInt(ptr, numSamples) || *ptr || numSamples < 1)
                fail("Invalid number of samples '%s'!", argv[i]);
        }
        else if (parseLiteral(ptr, "--sort="))
        {
            int value = 0;
            if (!parseInt(ptr, value) || *ptr || value < 0 || value > 1)
                fail("Invalid ray sorting enable/disable '%s'!", argv[i]);
            sortRays = (value != 0);
        }
        else if (parseLiteral(ptr, "--warmup-repeats="))
        {
            if (!parseInt(ptr, warmupRepeats) || *ptr || warmupRepeats < 0)
                fail("Invalid number of warmup repeats '%s'!", argv[i]);
        }
        else if (parseLiteral(ptr, "--measure-repeats="))
        {
            if (!parseInt(ptr, measureRepeats) || *ptr || measureRepeats < 1)
                fail("Invalid number of measurement repeats '%s'!", argv[i]);
        }
        else
        {
            fail("Invalid option '%s'!", argv[i]);
        }
    }

    // Show help.

    if (showHelp)
    {
        printf("%s", s_commandHelpText);
        exitCode = 1;
        return;
    }

    // Validate options.
    if (!meshFile.getLength())
        fail("Mesh file (--mesh) not specified!");
    
    if (!cameras.getSize())
        fail("No camera signatures (--camera) specified!");

    // Run modeBenchmark.
    runBenchmark(meshFile, cameras, sbvhAlpha, aoRadius, numSamples, sortRays, warmupRepeats, measureRepeats);

}

//------------------------------------------------------------------------

int main(int argc, char* argv[])
{
    // Store arguments.
    FW::argc = argc;
    FW::argv = argv;
    
    FW::init();
    
    return FW::exitCode;
}