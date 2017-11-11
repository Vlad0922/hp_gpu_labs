#include <cuda.h>

#include "matrix.h"

__global__
void convolve(const float *A, const float *B, float *dst, const int matrix_size, const int kernel_size)
{
    const int HM = (kernel_size - 1)/2;
    const int shared_size = blockDim.x + HM*2;

    const int block_area = blockDim.x*blockDim.y;
    const int shared_area = shared_size*shared_size;
    const int kernel_area = kernel_size*kernel_size;

    extern __shared__ float s[];

    float *s_kernel = s;
    float *s_matrix = &s[kernel_area];

    int t_idx = threadIdx.y*blockDim.x + threadIdx.x;

    int x_offset = blockIdx.x*blockDim.x;
    int y_offset = blockIdx.y*blockDim.y;

    int a_x = x_offset + threadIdx.x;
    int a_y = y_offset + threadIdx.y;

    // well...you know, kernel can be bigger than block
    for(int k = t_idx; k < kernel_area; k += block_area)
    {
        s_kernel[k] = B[k];
    }

    // copy current part of matrix from global memory to local
    // the function is a bit complicated becayse I want to evade if and pad with zeros
    // So matrix in shared memory is bigger then original
    for(int k = t_idx; k < shared_area; k += block_area)
    {
        int s_x = k % shared_size;
        int s_y = k / shared_size;

        int m_x = s_x - HM + x_offset;
        int m_y = s_y - HM + y_offset;

        if(m_x >= 0 && m_x < matrix_size && m_y >= 0 && m_y < matrix_size)
        {
            s_matrix[k] = A[m_y*matrix_size + m_x];
        }
        else
        {
            s_matrix[k] = 0; 
        }

    }

    __syncthreads();

    if(a_x < matrix_size && a_y < matrix_size)
    {
        float res = 0.;

        for(int k = 0; k < kernel_size; ++k)
        {
            int x = threadIdx.x + k;
            for(int l = 0; l < kernel_size; ++l)
            {
                int y = threadIdx.y + l;

                int ker_x = k;
                int ker_y = l;

                res += s_matrix[y*shared_size + x]*s_kernel[ker_y*kernel_size + ker_x];
            }
        }

        dst[a_y*matrix_size + a_x] = res;
    }
}

SquareMatrix convolve_with_cuda(const SquareMatrix &A, const SquareMatrix &B)
{
    const size_t BLOCK_X = 16;
    const size_t BLOCK_Y = 16;

    const size_t GRID_X = A.size()/BLOCK_X + int( (A.size() % BLOCK_X) != 0);
    const size_t GRID_Y = A.size()/BLOCK_Y + int( (A.size() % BLOCK_Y) != 0);

    const size_t HM = (B.size() - 1)/2;
    const size_t shared_memsize = ((BLOCK_X + 2*HM)*(BLOCK_Y+2*HM) + B.size()*B.size())*sizeof(float);
    
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
