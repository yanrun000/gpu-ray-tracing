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

#include "io/File.hh"

using namespace FW;

//------------------------------------------------------------------------

FileRO::FileRO(const String& name)
:   m_name          (name),
    m_file          (NULL)
{
    // Open.
    
    m_file = fopen(name.getPtr(), "r");
    
    if(m_file == NULL)
        printf("Cannot open file '%s' for reading!\n", m_name.getPtr());
}

//------------------------------------------------------------------------

FileRO::~FileRO(void)
{
    if (!m_file)
        return;

    fclose(m_file);
}

//------------------------------------------------------------------------

int FileRO::read(void* ptr, int size)
{
    if(m_file == NULL)
        fail("File %s is not opened!", m_name.getPtr());
    
    int numBytes = fread(ptr, sizeof(char), size, m_file);
    
    if(ferror(m_file))
        fail("Error reading file %s", m_name.getPtr());
        
    return numBytes;
}

//------------------------------------------------------------------------

FileWR::FileWR(const String& name)
:   m_name          (name),
    m_file          (NULL)
{
    // Open.
    
    m_file = fopen(name.getPtr(), "w");
    
    if(m_file == NULL)
        fail("Cannot open file '%s' for writing!", m_name.getPtr());
}

//------------------------------------------------------------------------

FileWR::~FileWR(void)
{
    if (!m_file)
        return;

    if(m_file) fclose(m_file);
}

//------------------------------------------------------------------------

void FileWR::write(const void* ptr, int size)
{
    if(m_file == NULL)
        fail("File %s is not opened!", m_name.getPtr());
    
    fwrite(ptr, sizeof(char), size, m_file);
    
    if(ferror(m_file))
        fail("Error writing file %s", m_name.getPtr());
}

void FileWR::close(void)
{
    if(m_file)
        fclose(m_file);
}

//------------------------------------------------------------------------
