#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>


template <typename scalar_t>
__device__ __forceinline__ void AggregatePath(
		scalar_t* Output, 
		const scalar_t* FourVec, 
		const scalar_t* Selector)
{

	(*Output) += (*FourVec)*(*Selector);  
}

template <typename scalar_t>
__global__ void SelectorKernel(
	const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> PTH, 
	const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> FV, 
	torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> Pmu,
	const size_t cmbi_l, 
	const size_t nodes)
{
	const int nd = blockIdx.y;
	const int index = blockIdx.x*blockDim.x + threadIdx.x;
	if (index < cmbi_l)
	{
		for (unsigned int i = 0; i < nodes; i++)
		{
			AggregatePath(&(Pmu[index][nd]), &(FV[i][nd]), &(PTH[index][i])); 
		}
	}
}

torch::Tensor PathVectorGPU(torch::Tensor AdjMatrix, torch::Tensor FourVector)
{
	torch::TensorOptions options = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA);
	const int cmbi_l = AdjMatrix.size(0);
	const int nodes = AdjMatrix.size(1);
	const int threads = 1024;
	const dim3 blocks((cmbi_l + threads -1) / threads, 4);
	
	torch::Tensor Pmu = torch::zeros({cmbi_l, 4}, options);
	FourVector = FourVector.to(options);
	AdjMatrix = AdjMatrix.to(options);

	AT_DISPATCH_FLOATING_TYPES(torch::kFloat, "SelectorKernel", ([&]
	{
		SelectorKernel<scalar_t><<<blocks, threads>>>(
			AdjMatrix.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(), 
			FourVector.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
			Pmu.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
			cmbi_l, nodes
		);
	}));
	return Pmu;
}


template <typename scalar_t>
__global__ void NodeSelectorKernel(
	const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> PTH, 
	const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> FV, 
	const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> NodeIndex, 
	torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> Pmu,
	const size_t cmbi_l, 
	const size_t nodes)
{
	const int adj = blockIdx.y*blockDim.y + threadIdx.y;
	const int index = blockIdx.x*blockDim.x + threadIdx.x;
	
	if (index < cmbi_l)
	{
		for (unsigned int i = 0; i < nodes; i++)
		{
			//const int edge_index = index - NodeIndex[index][0]*(PTH.size(0)-1) + i; 
			//AggregatePath(&(Pmu[index][nd]), &(FV[edge_index][nd]), &(PTH[adj_index][i])); 
			
			printf("%d %d %d\n", index, PTH[adj][i], adj); 
			Pmu[index][0] += PTH[adj][i];
		}
	}
}

torch::Tensor IncomingEdgeVectorGPU(torch::Tensor AdjMatrix, torch::Tensor IncomingEdges, torch::Tensor Index)
{

	torch::TensorOptions options = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA); 
	const int edges = IncomingEdges.size(0); 
	const int adj = AdjMatrix.size(0);
	const int nodes = AdjMatrix.size(1);

	torch::Tensor Pmu = torch::zeros({adj*nodes, 4}, options); 
	const int threads = 1024;
	const dim3 blocks((Pmu.size(0) + threads -1) / threads, 4);

	Index = Index.to(options);
	IncomingEdges = IncomingEdges.to(options);
	AdjMatrix = IncomingEdges.to(options);
	AT_DISPATCH_FLOATING_TYPES(torch::kFloat, "NodeSelectorKernel", ([&]
	{
		NodeSelectorKernel<scalar_t><<<blocks, threads>>>(
			AdjMatrix.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(), 
			IncomingEdges.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
			Index.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
			Pmu.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
			Pmu.size(0), nodes
		);
	})); 
	return Pmu; 

}
