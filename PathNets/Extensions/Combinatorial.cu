#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

template <typename scalar_t>
__device__ __forceinline__ scalar_t VecCalc(scalar_t fv, scalar_t sw)
{
	return fv*sw; 
}


template <typename scalar_t>
__global__ void MassKernel(
	scalar_t* __restrict__ Mass,
	const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> PTH, 
	const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> FV, 
	torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> Pmu,
	const size_t cmbi_l, 
	const size_t nodes)
{
	const int nd = blockIdx.y;
	const int index = blockIdx.x*blockDim.x + threadIdx.x;
	if (index < nodes)
	{
		printf("%d, %d\n", nd, index);
		Pmu[nd][0] += FV[index][0];
		Pmu[nd][1] += FV[index][1];
		Pmu[nd][2] += FV[index][2];
		Pmu[nd][3] += FV[index][3];
	}

}
torch::Tensor PathMassGPU(torch::Tensor AdjMatrix, torch::Tensor FourVector)
{
	torch::TensorOptions options = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA);
	const int cmbi_l = AdjMatrix.size(0);
	const int nodes = AdjMatrix.size(1);
	std::cout << nodes << " " << cmbi_l << std::endl;
	const int threads = nodes;
	const dim3 blocks((nodes + threads -1) / threads, cmbi_l);
	
	torch::Tensor Mass = torch::zeros({cmbi_l, 1}, options);
	torch::Tensor Pmu = torch::zeros({cmbi_l, 4}, options);
	FourVector = FourVector.to(options);
	AdjMatrix = AdjMatrix.to(options);

	AT_DISPATCH_FLOATING_TYPES(torch::kFloat, "MassKernel", ([&]
	{
		MassKernel<scalar_t><<<blocks, threads>>>(
			Mass.data_ptr<scalar_t>(),
			AdjMatrix.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(), 
			FourVector.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
			Pmu.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
			cmbi_l, nodes
		);
	}));
	return Pmu;

}
