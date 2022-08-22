#include <torch/extension.h>
#include <vector>

// CUDA forward declaration
torch::Tensor PathMassGPU(torch::Tensor AdjMatrix, torch::Tensor FourVector); 
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), "#x must be on CUDA")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), "#x must be contiguous") 
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor PathMassCUDA(torch::Tensor AdjMatrix, torch::Tensor FourVector)
{
	CHECK_INPUT(AdjMatrix); 
	CHECK_INPUT(FourVector);
	
	return PathMassGPU(AdjMatrix, FourVector);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("PathMassCUDA", &PathMassCUDA, "Invariant mass of path given the four vectors");
}
