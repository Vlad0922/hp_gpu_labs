

__global__ void convolve(const float *A, const float *B, float *dst, const int kernel_size)
{

}


SquareMatrix convolve_with_cuda(const SquareMatrix &A, const SquareMatrix &B)
{
	SquareMatrix C(A.size());

	float *dev_A;
	float *dev_kernel;
	float *dev_result;
	int kernel_size;	

	cudaMalloc((void **)&dev_result, A.size()*A.size()*sizeof(float));
	cudaMalloc((void **)&dev_A, A.size()*A.size()*sizeof(float));
	cudaMalloc((void **)&dev_kernel, B.size()*B.size()*sizeof(float));	

	cudaMemcpy(dev_A, A.data(), A.size()*A.size()*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_kernel, B.data(), B.size()*B.size()*sizeof(float), cudaMemcpyHostToDevice);

	convolve<<<8,8>>>(dev_A, dev_kernel, dev_result)

	cudaMemcpy(C.data(), dev_result, C.size()*C.size()*sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(dev_A);
	cudaFree(dev_kernel);
	cudaFree(dev_result);


	return C;
}