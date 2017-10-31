#include <iostream>
#include <fstream>
#include <sstream>
#include <tuple>

#include <assert.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "matrix.h"

const size_t BLOCK_X = 16;
const size_t BLOCK_Y = 16;

__global__
void convolve(const float *A, const float *B, float *dst, const int matrix_size, const int kernel_size)
{
    __shared__ float s[BLOCK_X*BLOCK_Y];

    int i = (blockIdx.x*blockDim.x) + threadIdx.x;
    int j = (blockIdx.y*blockDim.y) + threadIdx.y;

    if(i < matrix_size && j < matrix_size)
    {
        float res = 0.;
        const int HM = (kernel_size - 1)/2;

        for(int k = -HM; k <= HM; ++k)
        {
            for(int l = -HM; l <= HM; ++l)
            {
                int x = i + k;
                int y = j + l;

                int ker_x = k + HM;
                int ker_y = l + HM;

                if( x >= 0 && x < matrix_size && y >= 0 && y < matrix_size)
                {
                    res += A[y*matrix_size + x]*B[ker_y*kernel_size + ker_x];
                }
            }
        }

        s[threadIdx.y*BLOCK_X + threadIdx.x] = res;
    }

    __syncthreads();

    if(i < matrix_size && j < matrix_size)
    {
        dst[j*matrix_size+i] = s[threadIdx.y*BLOCK_X + threadIdx.x];
    }
}


SquareMatrix convolve_with_cuda(const SquareMatrix &A, const SquareMatrix &B)
{
    const size_t GRID_X = A.size()/BLOCK_X + int( (A.size() % BLOCK_X) != 0);
    const size_t GRID_Y = A.size()/BLOCK_Y + int( (A.size() % BLOCK_Y) != 0);

    const dim3 blockSize(BLOCK_X, BLOCK_Y, 1);  
    const dim3 gridSize(GRID_X, GRID_Y, 1); 

    SquareMatrix C(A.size());

    float *dev_A;
    float *dev_kernel;
    float *dev_result;

    cudaMalloc((void **)&dev_result, A.size()*A.size()*sizeof(float));
    cudaMalloc((void **)&dev_A, A.size()*A.size()*sizeof(float));
    cudaMalloc((void **)&dev_kernel, B.size()*B.size()*sizeof(float));  

    cudaMemcpy(dev_A, A.data(), A.size()*A.size()*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_kernel, B.data(), B.size()*B.size()*sizeof(float), cudaMemcpyHostToDevice);

    convolve<<<gridSize, blockSize>>>(dev_A, dev_kernel, dev_result, A.size(), B.size());

    cudaMemcpy(C.data(), dev_result, C.size()*C.size()*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dev_A);
    cudaFree(dev_kernel);
    cudaFree(dev_result);

    return C;
}

const char *INPUT_FNAME  = "input.txt";
const char *OUTPUT_FNAME = "output.txt";

void read_from_fstream(std::ifstream &input, float *arr, const size_t size)
{
    float val;
    for(size_t i = 0; i < size; ++i)
    {
        for(size_t j = 0; j < size; ++j)
        {
            input >> val;
            arr[i*size + j] = val;
        }
    }
}

std::tuple<SquareMatrix, SquareMatrix> read_data(const char *fname)
{
    std::ifstream input_file(fname);

    if(!input_file.is_open())
    {
        std::cerr << "Cannot open " << fname << "!\n";
        exit(1);
    }

    size_t N, M;

    input_file >> N >> M;

    SquareMatrix A(N);
    SquareMatrix B(M);

    read_from_fstream(input_file, A.data(), N);
    read_from_fstream(input_file, B.data(), M);

    return std::make_tuple(A, B);
}

void print_matrix(const SquareMatrix &m)
{
    for(size_t i = 0; i < m.size(); ++i)
    {
        for(size_t j = 0; j < m.size(); ++j)
        {
            std::cout << m[i][j] << ' ';
        }
        std::cout << '\n';
    }
}


float calc_cell_convolve(const SquareMatrix &A, const SquareMatrix &B, int i, int j)
{
    float val = 0;
    int HM = (B.size() - 1)/2;

    for(int k = -HM; k <= HM; ++k)
    {
        for(int l = -HM; l <= HM; ++l)
        {
            val += A.get_val(i+k, j+l)*B.get_val(k+HM, l+HM);
        }
    }

    return val;
}


// simple function to check the result
SquareMatrix convolve(const SquareMatrix &A, const SquareMatrix &B)
{
    SquareMatrix C(A.size());

    for(int i = 0; i < A.size(); ++i)
    {
        for(int j = 0; j < A.size(); ++j)
        {
            C[i][j] = calc_cell_convolve(A, B, i, j);
        }
    }

    return C;
}


void write_data(const char *fname, const SquareMatrix &m)
{
    std::ofstream out_file(fname);

    if(!out_file.is_open())
    {        
        std::cerr << "Cannot open " << fname << "!\n";
        return;
    }    

    for(size_t i = 0; i < m.size(); ++i)
    {
        for(size_t j = 0; j < m.size(); ++j)
        {
            out_file << m[j][i] << ' ';
        }
        out_file << '\n';
    }
}


void run_test()
{
    {
        SquareMatrix A(1024);
        SquareMatrix B(3);

        A.fill(1.);
        B.fill(1.);

        SquareMatrix correct = convolve(A, B);
        SquareMatrix cuda_res = convolve_with_cuda(A, B);

        assert(correct == cuda_res && "1st test didn't pass");
    }

    {
        SquareMatrix A(1024);
        SquareMatrix B(9);

        A.fill(1.);
        B.fill(1.);

        SquareMatrix correct = convolve(A, B);
        SquareMatrix cuda_res = convolve_with_cuda(A, B);

        assert(correct == cuda_res && "2nd test didn't pass");
    }

    {
        SquareMatrix A(1);
        SquareMatrix B(9);

        A.fill(1.);
        B.fill(1.);

        SquareMatrix correct = convolve(A, B);
        SquareMatrix cuda_res = convolve_with_cuda(A, B);

        assert(correct == cuda_res && "3rd test didn't pass");
    }

    {
        SquareMatrix A(31);
        SquareMatrix B(9);

        A.fill(1.);
        B.fill(1.);

        SquareMatrix correct = convolve(A, B);
        SquareMatrix cuda_res = convolve_with_cuda(A, B);

        assert(correct == cuda_res && "4th test didn't pass");
    }

    {
        SquareMatrix A(1023);
        SquareMatrix B(9);

        A.fill(1.);
        B.fill(1.);

        SquareMatrix correct = convolve(A, B);
        SquareMatrix cuda_res = convolve_with_cuda(A, B);

        assert(correct == cuda_res && "5th test didn't pass");
    }

    std::cout << "All tests passed!\n";
}


int main(int argc, char **argv)
{
    if(argc == 2 && strcmp(argv[1], "test") == 0)
    {
        run_test();
    }
    else
    {
        SquareMatrix A, B;
        std::tie(A, B) = read_data(INPUT_FNAME);

        SquareMatrix correct = convolve(A, B);
        SquareMatrix cuda_res = convolve_with_cuda(A, B);
        
        if(correct == cuda_res)
        {
            std::cout << "Ok!\n";
            
        }
        else
        {
            std::cout << "Error!\n";
        }

        write_data(OUTPUT_FNAME, cuda_res);
    }


    return 0;
}
