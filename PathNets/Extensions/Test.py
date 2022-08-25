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

#
#from PathNetOptimizerCUDA import * 
#
#n = 20
#AdjM = PathCombinatorial(n, n, "cuda")
#TestVect = torch.tensor([[i+1, i+1, i+1, i+1] for i in range(n)], device = "cuda")
#new = torch.cat([TestVect]*n, dim = 0)
#index = torch.tensor([[i] for i in range(n) for j in range(n)], device = "cuda")
#
#import time
#ts_0 = time.time()
#l = []
#for i in range(n):
#    for j in AdjM:
#        l.append(sum(new[i*len(j):(i+1)*len(j)][j == 1]).tolist())
#te_0 = time.time()
#
#
#
#    
#
#print("------")
#ts_1 = time.time()
#v = EdgeVector(AdjM, new, index)
#te_1 = time.time()
#
#print("cpu", te_0 - ts_0, "cuda", te_1 - ts_1)
#
#for j, i in zip(torch.as_tensor(l), v):
#    j, i = j.tolist(), i.tolist()
#    assert i == j





from PathNetOptimizerCUDA import * 

nodes = 4
mx = 2
#print(PathCombinatorial(n, mx))


def Combinatorial(n, k, msk, t = [], v = [], num = 0):

    if n == 0:
        t += [torch.tensor(num).unsqueeze(-1).bitwise_and(msk).ne(0).to(dtype = int)]
        v += [num]
        return t, v

    if n-1 >= k:
        t, v = Combinatorial(n-1, k, msk, t, v, num)
    if k > 0:
        t, v = Combinatorial(n-1, k -1, msk, t, v, num | ( 1 << (n-1)))

    return t, v


msk = torch.pow(2, torch.arange(nodes))
for i in range(1, mx+1):
    out, num = Combinatorial(nodes, i, msk)

def fact(n):
    if n == 0:
        return 1
    return n*fact(n-1)

cmb = 0
for i in range(1, mx+1):
    cmb += fact(nodes) / ( fact(nodes - i) * fact(i) )



for t, z in zip(out, num):
    print(t, z)

print("----")
mkr = []
nkr = []



for n in range(nodes):
    out = msk[n]
    for l in range(nodes):
        if n > l:
            continue
        out = msk[n] + msk[l]
        print([torch.tensor(int(out)).unsqueeze(-1).bitwise_and(msk).ne(0).to(dtype = int)])


exit()
print(len(out), len(mkr), len(nkr), cmb)

for i in out:
    f = False
    it = -1
    for j in mkr:
        it += 1
        if torch.sum(torch.eq(i, j)) != len(j):
            continue
        f = True 
        j = mkr.pop(it)
        break

    if f == False:
        print(i, j)
    if f:
        print("--->", len(mkr))




