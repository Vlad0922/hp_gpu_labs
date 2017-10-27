NVCC=nvcc
NVCC_OPTS=-O3 -arch=sm_20 -Xcompiler -Wall -Xcompiler -Wextra -m64 -std=c++11

all:
	$(NVCC) $(NVCC_OPTS) convolve.cu -o lab_1
clean:
	rm lab_1 