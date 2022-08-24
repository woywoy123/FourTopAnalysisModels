#include <torch/extension.h>
#include <vector>
#include <iostream>

using namespace torch::indexing; 

void Combinatorial(int n, int k, int num, std::vector<torch::Tensor>* out, torch::TensorOptions options, torch::Tensor msk)
{ 
	if (n == 0)
	{
		if (k == 0)
		{ 
			torch::Tensor tmp = torch::tensor(num, options);
			tmp = tmp.unsqueeze(-1).bitwise_and(msk).ne(0).to(torch::kInt);
			out -> push_back(tmp);
		}
		return; 
	}
	if (n -1 >= k){ Combinatorial(n-1, k, num, out, options, msk); }
	if (k > 0){ Combinatorial(n -1, k -1, num | ( 1 << (n -1)), out, options, msk); }
}


torch::Tensor PathCombinatorial(int n, unsigned int max, std::string device)
{
	torch::TensorOptions options = torch::TensorOptions();
	if (device == "cuda"){options = options.device(torch::kCUDA);}
	torch::Tensor msk = torch::pow(2, torch::arange(n, options));

	std::vector<torch::Tensor> nodes; 
	for (unsigned int k = 1; k < max +1; ++k)
	{
		Combinatorial(n, k, 0, &nodes, options, msk);
	}
	
	return torch::stack(nodes).to(options);
}

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

torch::Tensor PathVectorCPU(torch::Tensor AdjMatrix, torch::Tensor FourVector)
{
	std::vector<torch::Tensor> MassCombi; 
	for (unsigned int i = 0; i < AdjMatrix.sizes()[0]; ++i)
	{
		torch::Tensor x = torch::sum(FourVector.index({AdjMatrix[i] == 1}), {0});
		MassCombi.push_back(x);
	}
	return torch::stack(MassCombi);
}

torch::Tensor PathMassCPU(torch::Tensor AdjMatrix, torch::Tensor FourVector)
{
	return MassFromPxPyPzE(PathVectorCPU(AdjMatrix, FourVector)); 
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("PathCombinatorial", &PathCombinatorial, "Path Combinatorial");
  m.def("PathVector", &PathVectorCPU, "Summation of four vectors");
  m.def("PathMass", &PathMassCPU, "Invariant Mass"); 
}
