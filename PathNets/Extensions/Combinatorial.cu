#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>




template <typename scalar_t>
__global__ void Kernel(
		const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> PTH, 
		const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> FV, 




torch::Tensor PathMassGPU(torch::Tensor AdjMatrix, torch::Tensor FourVector)
{
	const int l = AdjMatrix.size(0);
	





	return AdjMatrix;

}
