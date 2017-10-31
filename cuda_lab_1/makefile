NVCC=nvcc
NVCC_OPTS=-O3 -Xcompiler -Wall -Xcompiler -Wextra -std=c++11

all:
	$(NVCC) $(NVCC_OPTS) src/convolve.cu -o bin/lab_1
clean:
	rm bin/*