#include <torch/extension.h>
#include <vector>
#include <iostream>


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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("PathCombinatorial", &PathCombinatorial, "Path Combinatorial");
}
