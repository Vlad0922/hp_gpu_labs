#include <vector>

const int BLOCK_X = 256;

std::vector<float> calc_prefix_cpu(const std::vector<float> &data)
{
    std::vector<float> res(data.size());

    res[0] = data[0];
    for(size_t i = 1; i < data.size(); ++i)
    {
        res[i] = res[i-1] + data[i];
    }

    return res;
}

__global__ void cuda_prefixsum(float *input, float *output, int sz) 
{
    __shared__ float s[BLOCK_X*2];

    unsigned int tidx = threadIdx.x;

    //load data into the shared memory
    int left_idx = 2*blockIdx.x*blockDim.x + tidx;
    int right_idx = left_idx + blockDim.x ;

    if(left_idx < sz)
    {
        s[tidx] = input[left_idx];
    }
    else
    {
        s[tidx] = 0.;
    }

    if(right_idx < sz)
    {
        s[tidx + blockDim.x] = input[right_idx];
    }
    else
    {
        s[tidx + blockDim.x] = 0.;
    }

    __syncthreads();

    // forward pass
    for (int stride = 1; stride <= blockDim.x; stride <<= 1) 
    {
        int idx = (threadIdx.x + 1)*stride*2 - 1;
        if (idx < 2*blockDim.x)
        {
            s[idx] += s[idx - stride];
        }

        __syncthreads();
    }

    // backward pass
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1)
    {
        int idx = (threadIdx.x + 1)*stride*2 - 1;
        if (idx + stride < 2*blockDim.x) 
        {
            s[idx + stride] += s[idx];
        }

        __syncthreads();
    }

    if (left_idx < sz)
    {
        output[left_idx] = s[tidx];
    }
    if (right_idx < sz)
    {
        output[right_idx] = s[tidx + blockDim.x];
    }
}

__global__ void aggregate(float *input, float *output, int sz) 
{
    int tidx = threadIdx.x;
    int dest_idx = (tidx + 1)*BLOCK_X*2 - 1;

    if (dest_idx < sz) 
    {
        output[tidx] = input[dest_idx];
    }
}


__global__ void collect_sums(float *input, float *output, int sz) 
{
    int dest_idx = threadIdx.x + blockDim.x*(blockIdx.x + 1);

    if (dest_idx < sz) 
    {
        output[dest_idx] += input[blockIdx.x];
    }
}

std::vector<float> calc_prefix_cuda(const std::vector<float> &data)
{
    const size_t num_elems = data.size();
    const size_t grid_x = (num_elems - 1)/(BLOCK_X*2) + 1;

    std::vector<float> res(num_elems);

    float *dev_data;
    float *dev_buffer;
    float *dev_aggregate;
    float *dev_res;

    cudaMalloc((void **)&dev_data,      num_elems*sizeof(float));
    cudaMalloc((void **)&dev_buffer,    num_elems*sizeof(float));  
    cudaMalloc((void **)&dev_aggregate, grid_x*sizeof(float));
    cudaMalloc((void **)&dev_res,       grid_x*sizeof(float));

    cudaMemset(dev_buffer, 0., num_elems*sizeof(float));
    cudaMemcpy(dev_data, data.data(), num_elems*sizeof(float), cudaMemcpyHostToDevice);

    cuda_prefixsum<<<grid_x, BLOCK_X>>>(dev_data, dev_buffer, num_elems);
    aggregate<<<1, grid_x>>>(dev_buffer,dev_aggregate, num_elems);
    cuda_prefixsum<<<1, grid_x>>>(dev_aggregate, dev_res, grid_x);
    collect_sums<<<grid_x, 2*BLOCK_X>>> (dev_res, dev_buffer, num_elems); 

    cudaMemcpy(res.data(), dev_buffer, num_elems*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dev_data);
    cudaFree(dev_buffer);
    cudaFree(dev_aggregate);
    cudaFree(dev_res);

    return res;
}