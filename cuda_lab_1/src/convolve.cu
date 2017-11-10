#include <iostream>
#include <fstream>
#include <sstream>
#include <tuple>
#include <vector>

#include <assert.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "matrix.h"

const size_t BLOCK_X = 16;
const size_t BLOCK_Y = 16;

__device__
void fill_border(float *dst, const int size, const int HM, const float val)
{
    for(int i = 0; i < size; ++i)
    {
        for(int j = 0; j < HM; ++j)
        {
            dst[j*size + i] = val;
        }

        for(int j = size - HM; j < size; ++j)
        {
            dst[j*size + i] = val;
        }
    }

    for(int j = HM; j < size - HM; ++j)
    {
        for(int i = 0; i < HM; ++i)
        {
            dst[j*size + i] = val;
        }

        for(int i = size - HM; i < size; ++i)
        {
            dst[j*size + i] = val;
        }
    }
}

__global__
void convolve(const float *A, const float *B, float *dst, const int matrix_size, const int kernel_size)
{
    const int HM = (kernel_size - 1)/2;

    extern __shared__ float s_kernel[];

    // float *s_matrix = &s[kernel_size*kernel_size];
    // float *s_kernel = s;

    int x_offset = blockIdx.x*blockDim.x;
    int y_offset = blockIdx.y*blockDim.y;

    int a_x = x_offset + threadIdx.x;
    int a_y = y_offset + threadIdx.y;

    if(a_x < matrix_size && a_y < matrix_size)
    {
        if(threadIdx.x < kernel_size && threadIdx.y < kernel_size)
        {
            int ker_idx = threadIdx.y*kernel_size + threadIdx.x;
            s_kernel[ker_idx] = B[ker_idx];
        }

        // s_matrix[threadIdx.y*BLOCK_X + threadIdx.x] = A[a_y*matrix_size + a_x];

        __syncthreads();

        float res = 0.;

        for(int k = -HM; k <= HM; ++k)
        {
            int x = a_x + k;
            if (x >= 0 && x < matrix_size)
            {
                for(int l = -HM; l <= HM; ++l)
                {
                    int y = a_y + l;

                    if(y >= 0 && y < matrix_size)
                    {                    
                        int ker_x = k + HM;
                        int ker_y = l + HM;

                        res += A[y*matrix_size + x]*s_kernel[ker_y*kernel_size + ker_x];
                    }
                }
            }
        }

        dst[a_y*matrix_size + a_x] = res;
    }
}

SquareMatrix convolve_with_cuda(const SquareMatrix &A, const SquareMatrix &B)
{
    const size_t GRID_X = A.size()/BLOCK_X + int( (A.size() % BLOCK_X) != 0);
    const size_t GRID_Y = A.size()/BLOCK_Y + int( (A.size() % BLOCK_Y) != 0);

    const size_t HM = (B.size() - 1)/2;
    // const size_t shared_memsize = ((BLOCK_X + 2*HM)*(BLOCK_Y+2*HM) + B.size()*B.size())*sizeof(float);
    const size_t shared_memsize = B.size()*B.size()*sizeof(float);

    const dim3 blockSize(BLOCK_X, BLOCK_Y, 1);  
    const dim3 gridSize(GRID_X, GRID_Y, 1); 

    SquareMatrix C(A.size());

    float *dev_A;
    float *dev_kernel;
    float *dev_result;

    cudaMalloc((void **)&dev_result, A.size()*A.size()*sizeof(float));
    cudaMalloc((void **)&dev_A,      A.size()*A.size()*sizeof(float));
    cudaMalloc((void **)&dev_kernel, B.size()*B.size()*sizeof(float));  

    cudaMemcpy(dev_A,       A.data(), A.size()*A.size()*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_kernel,  B.data(), B.size()*B.size()*sizeof(float), cudaMemcpyHostToDevice);

    convolve<<<gridSize, blockSize, shared_memsize>>>(dev_A, dev_kernel, dev_result, A.size(), B.size());

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
            out_file << m[i][j] << ' ';
        }
        out_file << '\n';
    }
}


void run_test()
{
    using test_val_t = std::tuple<SquareMatrix, SquareMatrix, const char *>;

    auto test_func = [](const SquareMatrix &A, const SquareMatrix &B)
                        {
                            SquareMatrix correct = convolve(A, B);
                            SquareMatrix cuda_res = convolve_with_cuda(A, B);

                            // bool res = correct == cuda_res;

                            // if(!res)
                            // {
                            //     print_matrix(correct);
                            //     std::cout << "******\n";
                            //     print_matrix(cuda_res);
                            // }

                            return correct == cuda_res;
                        };

    std::vector<test_val_t> test_vals = {
                                            std::make_tuple(SquareMatrix(5, 1.),    SquareMatrix(3, 1.), "1st"),
                                            std::make_tuple(SquareMatrix(1024, 1.), SquareMatrix(3, 1.), "2nd"),
                                            std::make_tuple(SquareMatrix(1024, 1.), SquareMatrix(9, 1.), "3rd"),
                                            std::make_tuple(SquareMatrix(1, 1.),    SquareMatrix(9, 1.), "4th"),
                                            std::make_tuple(SquareMatrix(31, 1.),   SquareMatrix(9, 1.), "5th"),
                                            std::make_tuple(SquareMatrix(1023, 1.), SquareMatrix(9, 1.), "6th")
                                        };

    for(const test_val_t &test : test_vals)
    {
        bool res = test_func(std::get<0>(test), std::get<1>(test));

        if(res)
        {
            std::cout << std::get<2>(test) << " is correct\n";
        }
        else
        {
            std::cout << std::get<2>(test) << " is incorrect\n";
            return;
        }
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
