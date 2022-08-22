import torch
from PathNetOptimizer import *
from PathNetOptimizerCUDA import * 

AdjM = PathCombinatorial(3, 3, "cpu")
TestVect = torch.Tensor([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]], device = "cpu")
print(AdjM)
print(TestVect)

for i in AdjM:
    print("-> ", i)
    
    print("-----")
    print(TestVect[i == 1])

print(PathMassCPU(AdjM, TestVect))

AdjM = PathCombinatorial(3, 3, "cuda")
TestVect = torch.tensor([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]], device = "cuda")
PathMassCUDA(AdjM, TestVect)
