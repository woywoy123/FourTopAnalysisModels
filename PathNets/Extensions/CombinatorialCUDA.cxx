#include <torch/extension.h>
#include <vector>

using namespace torch::indexing; 

torch::Tensor MassFromPxPyPzE(torch::Tensor v)
{
  v = v.pow(2);  
  v = v.view({-1, 4});
  
  torch::Tensor px = v.index({Slice(), Slice(0, 1, 1)}); 
  torch::Tensor py = v.index({Slice(), Slice(1, 2, 2)}); 
  torch::Tensor pz = v.index({Slice(), Slice(2, 3, 3)}); 
  torch::Tensor e = v.index({Slice(), Slice(3, 4, 4)}); 
  
  torch::Tensor s2 = e - px - py - pz;
  return torch::sqrt(s2.abs()); 
}

// CUDA forward declaration
torch::Tensor PathVectorGPU(torch::Tensor AdjMatrix, torch::Tensor FourVector); 
torch::Tensor IncomingEdgeVectorGPU(torch::Tensor AdjMatrix, torch::Tensor IncomingEdges, torch::Tensor Index);
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), "#x must be on CUDA")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), "#x must be contiguous") 
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor PathVectorCUDA(torch::Tensor AdjMatrix, torch::Tensor FourVector)
{
	CHECK_INPUT(AdjMatrix); 
	CHECK_INPUT(FourVector);
	
	return PathVectorGPU(AdjMatrix, FourVector);
}


torch::Tensor PathMassCUDA(torch::Tensor AdjMatrix, torch::Tensor FourVector)
{
	return MassFromPxPyPzE(PathVectorCUDA(AdjMatrix, FourVector)); 
}

torch::Tensor IncomingEdgeVectorCUDA(torch::Tensor AdjMatrix, torch::Tensor IncomingEdges, torch::Tensor Index)
{
	CHECK_INPUT(AdjMatrix);
	CHECK_INPUT(IncomingEdges);
	CHECK_INPUT(Index);
		
	return IncomingEdgeVectorGPU(AdjMatrix, IncomingEdges, Index);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("PathVector", &PathVectorCUDA, "Summation of four vectors");
  m.def("PathMass", &PathMassCUDA, "Invariant Mass"); 
  m.def("EdgeVector", &IncomingEdgeVectorCUDA, "Computes the aggregated vector for different combinatorial of incoming edges");
}
