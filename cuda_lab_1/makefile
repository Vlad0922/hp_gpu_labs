BIN_PATH="bin"

NVCC=nvcc
NVCC_OPTS=-O3 -Xcompiler -Wall -Xcompiler -Wextra -std=c++11

all:
	if [ ! -d $(BIN_PATH) ]; then	\
		mkdir $(BIN_PATH);			\
	fi								

	$(NVCC) $(NVCC_OPTS) src/convolve.cu -o $(BIN_PATH)/lab_1
clean:
	rm bin/*