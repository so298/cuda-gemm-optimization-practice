CXX := g++
NVCC := nvcc

CXXFLAGS = -O3 -std=c++17 -Wall
NVCCFLAGS = -gencode=arch=compute_80,code=sm_80 -O3 -std=c++17 -Xcompiler -Wall

CC_SOURCES = $(wildcard src/*.cc)
CU_SOURCES = $(wildcard src/*.cu)

CC_EXECUTABLES = $(CC_SOURCES:src/%.cc=exe/%)
CU_EXECUTABLES = $(CU_SOURCES:src/%.cu=exe/%)

all: $(CC_EXECUTABLES) $(CU_EXECUTABLES)

cpu: $(CC_EXECUTABLES)
gpu: $(CU_EXECUTABLES)

# a rule for .cc files
exe/%: src/%.cc
	@mkdir -p exe
	$(CXX) $(CXXFLAGS) $< -o $@

# a rule for .cu files
exe/%: src/%.cu
	@mkdir -p exe
	$(NVCC) $(NVCCFLAGS) $< -o $@

exe/99-cublas: src/99-cublas.cu
	@mkdir -p exe
	$(NVCC) $(NVCCFLAGS) $< -lcublas -o $@

clean:
	rm -f $(CC_EXECUTABLES) $(CU_EXECUTABLES)
