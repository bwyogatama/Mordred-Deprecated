
PAPI_INCLUDE = -I tools/papi/src #-I$(datadir) #-I$(testlibdir) #-I$(validationlibdir)
PAPILIB = tools/papi/src/libpapi.a
CUDALIBS = -lcudart -lcupti -lcuda
PAPI_INCLUDE += -I tools/pcm/src/ #-I$(PAPI_CUPTI_ROOT)/include
PAPILIB += -L tools/papi/src/libpfm4/lib -lpfm -L tools/pcm/build/lib -lpcm

CUDA_PATH       ?= /usr/local/cuda
CUDA_INC_PATH   ?= $(CUDA_PATH)/include
CUDA_BIN_PATH   ?= $(CUDA_PATH)/bin

NVCC = nvcc

#SM_TARGETS   = -gencode=arch=compute_52,code=\"sm_52,compute_52\" 
SM_DEF     = -DSM520

SM_TARGETS   = -gencode=arch=compute_70,code=\"sm_70,compute_70\" 
#SM_DEF     = -DSM700

#GENCODE_SM50    := -gencode arch=compute_52,code=sm_52
GENCODE_SM70    := -gencode arch=compute_70,code=sm_70
GENCODE_FLAGS   := $(GENCODE_SM70)

#NVCCFLAGS += --std=c++11 $(SM_DEF) -Xptxas="-dlcm=cg -v" -lineinfo -Xcudafe -\# 
NVCCFLAGS += --std=c++14 $(SM_DEF) -Xptxas="-dlcm=cg -v" -lineinfo -Xcudafe -\# 
OPENMPFLAGS = -Xcompiler -fopenmp -lgomp

SRC = src
BIN = bin
OBJ = obj
INC = includes

CUB_DIR = cub/
INCLUDES = -I$(CUB_DIR) -I$(CUB_DIR)test -I. -I$(INC)

CFLAGS = -O3 -march=native -std=c++14 -ffast-math
LDFLAGS = -ltbb
CINCLUDES = -I$(INC)
CXX = clang++

$(OBJ)/%.o: $(SRC)/%.cu
	$(NVCC) -lcurand $(SM_TARGETS) $(NVCCFLAGS) $(CPU_ARCH) $(INCLUDES) $(LIBS) $(PAPI_INCLUDE) -O3 -dc $< -o $@

$(BIN)/%: $(OBJ)/%.o
	$(NVCC) $(PAPILIB) $(CUDALIBS) -ltbb $(SM_TARGETS) -lcurand $^ -o $@

$(OBJ)/cpu/%.o: $(SRC)/cpu/%.cpp
	$(NVCC) -lcurand $(SM_TARGETS) $(NVCCFLAGS) $(CPU_ARCH) $(INCLUDES) $(LIBS) -O3 -dc $< -o $@

#$(CXX) $(CFLAGS) $(CINCLUDES) -c $< -o $@

$(BIN)/cpu/%: $(OBJ)/cpu/%.o
	$(NVCC) -ltbb $(SM_TARGETS) -lcurand $^ -o $@
	
#$(CXX) -ltbb $^ -o $@

$(OBJ)/%.o: $(SRC)/%.cpp
	$(NVCC) -lcurand $(SM_TARGETS) $(NVCCFLAGS) $(CPU_ARCH) $(INCLUDES) $(LIBS) -O3 -dc $< -o $@

$(OBJ)/gpudb/CostModel.o: $(SRC)/gpudb/CostModel.cu
	$(NVCC) -lcurand -ltbb $(SM_TARGETS) $(NVCCFLAGS) $(CPU_ARCH) $(INCLUDES) $(LIBS) -O3 -dc $< -o $@

$(OBJ)/gpudb/QueryOptimizer.o: $(SRC)/gpudb/QueryOptimizer.cu
	$(NVCC) -lcurand -ltbb $(SM_TARGETS) $(NVCCFLAGS) $(CPU_ARCH) $(INCLUDES) $(LIBS) -O3 -dc $< -o $@

$(OBJ)/gpudb/QueryProcessing.o: $(SRC)/gpudb/QueryProcessing.cu
	$(NVCC) -lcurand -ltbb $(SM_TARGETS) $(NVCCFLAGS) $(CPU_ARCH) $(INCLUDES) $(LIBS) -O3 -dc $< -o $@

$(OBJ)/gpudb/CPUGPUProcessing.o: $(SRC)/gpudb/CPUGPUProcessing2.cu
	$(NVCC) -lcurand -ltbb $(SM_TARGETS) $(NVCCFLAGS) $(CPU_ARCH) $(INCLUDES) $(LIBS) -O3 -dc $< -o $@

$(OBJ)/gpudb/CPUProcessing.o: $(SRC)/gpudb/CPUProcessing.cu
	$(NVCC) -lcurand -ltbb $(SM_TARGETS) $(NVCCFLAGS) $(CPU_ARCH) $(INCLUDES) $(LIBS) -O3 -dc $< -o $@

$(OBJ)/gpudb/CPUProcessingHE.o: $(SRC)/gpudb/CPUProcessingHE.cu
	$(NVCC) -lcurand -ltbb $(SM_TARGETS) $(NVCCFLAGS) $(CPU_ARCH) $(INCLUDES) $(LIBS) -O3 -dc $< -o $@

$(OBJ)/gpudb/main.o: $(SRC)/gpudb/main.cu
	$(NVCC) -lcurand -ltbb $(SM_TARGETS) $(NVCCFLAGS) $(CPU_ARCH) $(INCLUDES) $(LIBS) -O3 -dc $< -o $@

$(OBJ)/gpudb/maintraffic.o: $(SRC)/gpudb/maintraffic.cu
	$(NVCC) -lcurand -ltbb $(SM_TARGETS) $(NVCCFLAGS) $(CPU_ARCH) $(INCLUDES) $(LIBS) -O3 -dc $< -o $@

$(OBJ)/gpudb/ondemand.o: $(SRC)/gpudb/ondemand.cu
	$(NVCC) -lcurand -ltbb $(SM_TARGETS) $(NVCCFLAGS) $(CPU_ARCH) $(INCLUDES) $(LIBS) -O3 -dc $< -o $@

$(BIN)/gpudb/main: $(OBJ)/gpudb/main.o $(OBJ)/gpudb/CacheManager.o $(OBJ)/gpudb/QueryOptimizer.o $(OBJ)/gpudb/CPUProcessing.o $(OBJ)/gpudb/CPUProcessingHE.o $(OBJ)/gpudb/CPUGPUProcessing.o $(OBJ)/gpudb/QueryProcessing.o $(OBJ)/gpudb/CostModel.o
	$(NVCC) $(SM_TARGETS) $(CUDALIBS) -ltbb -lcurand $^ -o $@

$(BIN)/gpudb/maintraffic: $(OBJ)/gpudb/maintraffic.o $(OBJ)/gpudb/CacheManager.o $(OBJ)/gpudb/QueryOptimizer.o $(OBJ)/gpudb/CPUProcessing.o $(OBJ)/gpudb/CPUProcessingHE.o $(OBJ)/gpudb/CPUGPUProcessing.o $(OBJ)/gpudb/QueryProcessing.o $(OBJ)/gpudb/CostModel.o
	$(NVCC) $(SM_TARGETS) $(CUDALIBS) -ltbb -lcurand $^ -o $@

$(BIN)/gpudb/ondemand: $(OBJ)/gpudb/ondemand.o $(OBJ)/gpudb/CacheManager.o $(OBJ)/gpudb/QueryOptimizer.o $(OBJ)/gpudb/CPUProcessing.o $(OBJ)/gpudb/CPUProcessingHE.o$(OBJ)/gpudb/CPUGPUProcessing.o $(OBJ)/gpudb/QueryProcessing.o $(OBJ)/gpudb/CostModel.o
	$(NVCC) $(SM_TARGETS) $(CUDALIBS) -ltbb -lcurand $^ -o $@

sort: test/ssb/sort.c
	gcc -o sort $< -std=c99 

NVCC_VER=11.2
CUB_VER=1.8.0

ifeq ($(NVCC_VER),11.2)
setup:
	mkdir -p bin/ssb obj/ssb
	mkdir -p bin/ops obj/ops
	mkdir -p bin/cpu/ssb obj/cpu/ssb
	mkdir -p bin/gpudb obj/gpudb
else
setup:
	if [ ! -d "cub"  ]; then \
     wget https://github.com/NVlabs/cub/archive/$(CUB_VER).zip; \
     unzip $(CUB_VER).zip; \
     mv cub-$(CUB_VER) cub; \
     rm $(CUB_VER).zip; \
	fi
	mkdir -p bin/ssb obj/ssb
	mkdir -p bin/ops obj/ops
	mkdir -p bin/cpu/ssb obj/cpu/ssb
	mkdir -p bin/gpudb obj/gpudb
endif

clean:
	rm -rf bin/* obj/*
