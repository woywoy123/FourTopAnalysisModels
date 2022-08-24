import torch
from PathNetOptimizer import *

#AdjM = PathCombinatorial(3, 3, "cpu")
#TestVect = torch.Tensor([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]], device = "cpu")
#print(AdjM)
#print(TestVect)
#
#for i in AdjM:
#    print("-> ", i)
#    
#    print("-----")
#    print(TestVect[i == 1])
#
#print(PathVector(AdjM, TestVect))

AdjM = PathCombinatorial(3, 3, "cuda")
TestVect = torch.tensor([[1, 1, 1, 1], [6, 6, 6, 6], [3, 3, 3, 3]], device = "cuda")
print(AdjM)
print(TestVect)

print("-----")

from PathNetOptimizerCUDA import * 
print(PathVector(AdjM, TestVect))

print("-----")

new = torch.cat([TestVect]*3, dim = 0)
index = torch.tensor([[i] for i in range(3) for j in range(3)], device = "cuda")
print(new, new.shape)
print(index, index.shape)


for i in range(3):
    for j in AdjM:
        print(i, sum(new[i*len(j):(i+1)*len(j)][j == 1]))
    

print("------")
v = EdgeVector(AdjM, new, index)
print(v, v.shape)


