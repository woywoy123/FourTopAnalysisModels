import torch
from torch.nn import Sequential as Seq

import torch_geometric
from torch_geometric.nn import MessagePassing, Linear

from PathNetOptimizer import PathCombinatorial
from LorentzVector import *

def MakeMLP(lay):
    out = []
    for i in range(len(lay)-1):
        x1, x2 = lay[i], lay[i+1]
        out += [Linear(x1, x2)]
    return Seq(*out)

class PathNetsTruthJet(MessagePassing):

    def __init__(self):
        super().__init__(aggr = None, flow = "target_to_source") 

        self.O_from_top = None
        self.L_from_top = "CEL"
        self.C_from_top = True

        self.max = 2
        self._n = 0

        self._Node = MakeMLP([1, 6, 2])
        self._AdjMatrix = None

    def forward(self, i, edge_index, num_nodes, N_eta, N_energy, N_pT, N_phi):
        device = edge_index.device
        Pmu = torch.cat([N_pT, N_eta, N_phi, N_energy], dim = 1)
        Pmu = TensorToPxPyPzE(Pmu)





        if int(num_nodes) != self._n or self._AdjMatrix == None:
            self._n = int(num_nodes)
            self._AdjMatrix = PathCombinatorial(self._n, self.max, str(device).split(":")[0])
            self._AdjMatrix = self._AdjMatrix == 1
        
        Sum = torch.tensor((self._AdjMatrix.shape[0], 4), device = device)
        for i in range(len(self._AdjMatrix)):
            Sum[i, :] = Pmu[self._AdjMatrix[i]].sum(dim = 0)

         




        print(self._AdjMatrix.shape)
        




        self.O_from_top = self._Node(N_pT)


