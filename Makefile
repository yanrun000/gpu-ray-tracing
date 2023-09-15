# export LD_LIBRARY_PATH=/usr/local/cuda/lib/
#CUDADIR :=/usr/local/cuda
#CUDADIR :=/usr/local/cuda4.0/cuda
#CUDADIR :=/usr/local/cuda5.5
CUDADIR :=/usr/local/cuda-11.4
#CUDASDKDIR :=/home/cuda/SDK4.0
#CUDASDKDIR :=/home/workspace/NVIDIA_CUDA-7.5_Samples

CXX :=nvcc

TOPDIR :=.
SRCDIR :=$(TOPDIR)/src
OBJDIR :=$(TOPDIR)/obj

FWDIR := $(SRCDIR)/framework
BASEDIR := $(FWDIR)/base
T3DDIR := $(FWDIR)/3d
IODIR := $(FWDIR)/io
GPUDIR := $(FWDIR)/gpu
GUIDIR := $(FWDIR)/gui

RTDIR := $(SRCDIR)/rt
CUDIR := $(RTDIR)/cuda
BVHDIR := $(RTDIR)/bvh
RAYDIR := $(RTDIR)/ray
KERNELDIR := $(RTDIR)/kernels

#sources under framework directory
SRCS +=$(BASEDIR)/Defs.cc
SRCS +=$(BASEDIR)/Hash.cc
SRCS +=$(BASEDIR)/Main.cc
SRCS +=$(BASEDIR)/Math.cc
SRCS +=$(BASEDIR)/String.cc
SRCS +=$(BASEDIR)/Sort.cc
SRCS +=$(BASEDIR)/UnionFind.cc

SRCS +=$(T3DDIR)/Mesh.cc
SRCS +=$(T3DDIR)/CameraControls.cc

SRCS +=$(IODIR)/Stream.cc
SRCS +=$(IODIR)/File.cc
SRCS +=$(IODIR)/MeshWavefrontIO.cc

SRCS +=$(GPUDIR)/Buffer.cc
SRCS +=$(GPUDIR)/CudaModule.cc
SRCS +=$(GPUDIR)/CudaKernel.cc

SRCS +=$(GUIDIR)/Image.cc

#sources under rt directory
SRCS +=$(RTDIR)/App.cc
SRCS +=$(RTDIR)/Scene.cc
SRCS +=$(RTDIR)/Util.cc

SRCS +=$(CUDIR)/fetch_ray.cu#自己加用于取光线数据
SRCS +=$(CUDIR)/fetch_node.cu#自己加用于取节点数据
SRCS +=$(CUDIR)/Renderer.cc
SRCS +=$(CUDIR)/CudaBVH.cc
SRCS +=$(CUDIR)/CudaTracer.cc
SRCS +=$(CUDIR)/RendererKernels.cu

SRCS +=$(BVHDIR)/BVH.cc
SRCS +=$(BVHDIR)/BVHNode.cc
SRCS +=$(BVHDIR)/SplitBVHBuilder.cc

SRCS +=$(RAYDIR)/RayBuffer.cc
SRCS +=$(RAYDIR)/RayGen.cc
SRCS +=$(RAYDIR)/PixelTable.cc
SRCS +=$(RAYDIR)/RayGenKernels.cu
SRCS +=$(RAYDIR)/RayBufferKernels.cu

#SRCS +=$(KERNELDIR)/fermi_speculative_while_while.cu
#SRCS +=$(KERNELDIR)/kepler_speculative_while_while.cu
#SRCS +=$(KERNELDIR)/fermi_dynamic_fetch.cu
SRCS +=$(KERNELDIR)/kepler_dynamic_fetch.cu


# heh.
OBJS := $(SRCS)
OBJS := $(subst $(BASEDIR),$(OBJDIR), $(OBJS))
OBJS := $(subst $(T3DDIR),$(OBJDIR), $(OBJS))
OBJS := $(subst $(IODIR),$(OBJDIR), $(OBJS))
OBJS := $(subst $(GPUDIR),$(OBJDIR), $(OBJS))
OBJS := $(subst $(GUIDIR),$(OBJDIR), $(OBJS))
OBJS := $(subst $(CUDIR),$(OBJDIR), $(OBJS))
OBJS := $(subst $(BVHDIR),$(OBJDIR), $(OBJS))
OBJS := $(subst $(RAYDIR),$(OBJDIR), $(OBJS))
OBJS := $(subst $(KERNELDIR),$(OBJDIR), $(OBJS))
OBJS := $(subst $(RTDIR),$(OBJDIR), $(OBJS))
OBJS := $(subst $(SRCDIR),$(OBJDIR), $(OBJS))
OBJS := $(patsubst %.cc,%.o,  $(OBJS))
OBJS := $(patsubst %.cu,%.o,  $(OBJS))

all: $(OBJDIR) gpu-rt

#CXXFLAGS :=-pipe -O2 -march=native -ffast-math -fomit-frame-pointer -fno-rtti -fno-exceptions
CXXFLAGS +=-g
CXXFLAGS +=-Wall -Wextra -Wshadow -Wno-unused
#CXXFLAGS +=-MMD
#CXXFLAGS +=-DNDEBUG
# FreeGlut has trouble parsing Glut init strings but pretends it's fine. kludge.
#CXXFLAGS +=-DUSE_FREEGLUT

CXXFLAGS +=-I$(FWDIR) -I$(RTDIR)
#CXXFLAGS +=-I$(CUDADIR)/include -I$(CUDASDKDIR)/C/common/inc/
CXXFLAGS +=-I$(CUDADIR)/include -I$(CUDASDKDIR)/common/inc/
#LDFLAGS  +=-L$(CUDADIR)/lib64 -L$(CUDASDKDIR)/C/lib -L$(CUDASDKDIR)/C/common/lib/linux
LDFLAGS  +=-L$(CUDADIR)/lib64 -L$(CUDASDKDIR)/lib -L$(CUDASDKDIR)/C/common/lib/linux
#LDFLAGS  +=-lcutil_x86_64 -lcuda -lcudart -lGL -lGLU -lGLEW_x86_64 -lglut # snatched from cuda
LDFLAGS  +=-lcuda -lcudart -lGL -lGLU -lGLEW -lglut # snatched from cuda

## NVCC
# grabbed from cuda examples. quickfix.
# /usr/local/cuda/bin/nvcc    --compiler-options -fno-strict-aliasing  -I. -I/usr/local/cuda/include -I../../common/inc -DUNIX -O3   -o data/threadMigration.cubin -cubin threadMigration.cu
NVCC := $(CUDADIR)/bin/nvcc
#NVCCFLAGS :=-g -I$(CUDADIR)/include -I$(CUDASDKDIR)/C/common/inc/ -I$(FWDIR) -I$(RTDIR)
NVCCFLAGS :=-g -O2 -I$(CUDADIR)/include -I$(CUDASDKDIR)/common/inc/ -I$(FWDIR) -I$(RTDIR)
NVCCFLAGS +=--ptxas-options=-v -Xcompiler "-Wall,-Wno-unused"
NVCCFLAGS +=--use_fast_math -arch=sm_35
#NVCCNFLAGS_NATIVE :=-fno-strict-aliasing
#NVCCNFLAGS_NATIVE +=-fomit-frame-pointer -frename-registers -march=native
NVCCFLAGS +=$(foreach flag, $(NVCCNFLAGS_NATIVE), --compiler-options $(flag))


## rules
$(OBJDIR)/%.o : $(BASEDIR)/%.cc
#	$(COMPILE.cc) $< -o $@
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(OBJDIR)/%.o : $(T3DDIR)/%.cc
#	$(COMPILE.cc) $< -o $@
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(OBJDIR)/%.o : $(IODIR)/%.cc
#	$(COMPILE.cc) $< -o $@
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(OBJDIR)/%.o : $(GPUDIR)/%.cc
#	$(COMPILE.cc) $< -o $@
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(OBJDIR)/%.o : $(GUIDIR)/%.cc
#	$(COMPILE.cc) $< -o $@
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(OBJDIR)/%.o : $(RTDIR)/%.cc
#	$(COMPILE.cc) $< -o $@
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(OBJDIR)/%.o : $(BVHDIR)/%.cc
#	$(COMPILE.cc) $< -o $@
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(OBJDIR)/%.o : $(CUDIR)/%.cc
#	$(COMPILE.cc) $< -o $@
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(OBJDIR)/%.o : $(RAYDIR)/%.cc
#	$(COMPILE.cc) $< -o $@
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(OBJDIR)/%.o: $(RAYDIR)/%.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(OBJDIR)/%.o: $(KERNELDIR)/%.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(OBJDIR)/%.o: $(CUDIR)/%.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@


## targets
gpu-rt: $(OBJS)
	$(CXX) $^ -o $@ $(LDFLAGS)

clean: 
	rm -f gpu-rt
	rm -rf $(OBJDIR)

$(OBJDIR):
	mkdir -p $(OBJDIR)
		
-include $(OBJDIR)/*.d
