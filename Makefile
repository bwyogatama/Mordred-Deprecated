CUDA_PATH       ?= /usr/local/cuda
CUDA_INC_PATH   ?= $(CUDA_PATH)/include
CUDA_BIN_PATH   ?= $(CUDA_PATH)/bin

NVCC = nvcc

SM_TARGETS   = -gencode=arch=compute_52,code=\"sm_52,compute_52\" 
SM_DEF     = -DSM520

#SM_TARGETS   = -gencode=arch=compute_70,code=\"sm_70,compute_70\" 
#SM_DEF     = -DSM700

GENCODE_SM50    := -gencode arch=compute_52,code=sm_52
#GENCODE_SM70    := -gencode arch=compute_70,code=sm_70
GENCODE_FLAGS   := $(GENCODE_SM50)

#NVCCFLAGS += --std=c++11 $(SM_DEF) -Xptxas="-dlcm=cg -v" -lineinfo -Xcudafe -\# 
NVCCFLAGS += --std=c++14 $(SM_DEF) -Xptxas="-dlcm=cg -v" -lineinfo -Xcudafe -\# 

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
	$(NVCC) -lcurand $(SM_TARGETS) $(NVCCFLAGS) $(CPU_ARCH) $(INCLUDES) $(LIBS) -O3 -dc $< -o $@

$(BIN)/%: $(OBJ)/%.o
	$(NVCC) -ltbb $(SM_TARGETS) -lcurand $^ -o $@

$(OBJ)/cpu/%.o: $(SRC)/cpu/%.cpp
	$(CXX) $(CFLAGS) $(CINCLUDES) -c $< -o $@

$(BIN)/cpu/%: $(OBJ)/cpu/%.o
	$(CXX) -ltbb $^ -o $@


setup:
	if [ ! -d "cub"  ]; then \
    wget https://github.com/NVlabs/cub/archive/1.6.4.zip; \
    unzip 1.6.4.zip; \
    mv cub-1.6.4 cub; \
    rm 1.6.4.zip; \
	fi
	mkdir -p bin/ssb obj/ssb
	mkdir -p bin/ops obj/ops
	mkdir -p bin/gpudb obj/gpudb

clean:
	rm -rf bin/* obj/*
