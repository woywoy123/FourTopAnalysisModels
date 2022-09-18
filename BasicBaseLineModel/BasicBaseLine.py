import torch 
from torch_geometric.nn import MessagePassing
from torch.nn import Sequential as Seq, Linear, ReLU, Sigmoid, Tanh
from LorentzVector import *
from torch_geometric.utils import to_dense_adj

def MakeMLP(lay):
    out = []
    for i in range(len(lay)-1):
        x1, x2 = lay[i], lay[i+1]
        out += [Linear(x1, x2)]
    return Seq(*out)


class BasicBaseLine(MessagePassing):

    def __init__(self):
        super().__init__(aggr = None, flow = "target_to_source")
        self.O_edge = None
        self.L_edge = "CEL"
        self.C_edge = True

        end = 2048
        self._isEdge = Seq(Linear(3, end), Sigmoid(), Linear(end, 2))
        self._int = 0

    def forward(self, i, edge_index, N_pT, N_eta, N_phi, N_energy, N_mass):

        Pmu = torch.cat([N_pT, N_eta, N_phi, N_energy], dim = 1)
        Pmc = TensorToPxPyPzE(Pmu)
        mass, edge, edge_sc = self.propagate(edge_index, Pmc = Pmc, Pmu = Pmu, Mass = N_mass)

        if self._int == 0:
            self.O_edge = edge, edge_index
        
        # Continue here...
        self._it += 1 
        edge_index_new = edge_index.t().view(-1, 2)[edge == 1].t().view(2, -1)
        return self.forward(i, edge_index_new, N_pT, N_eta, N_phi, N_energy, mass)




    def message(self, edge_index, Pmc_i, Pmc_j, Pmu_i, Pmu_j, Mass_i, Mass_j):

        e_dr = TensorDeltaR(Pmu_i, Pmu_j)
        e_mass = MassFromPxPyPzE(Pmc_i + Pmc_j) / 1000
        edge = self._isEdge(torch.cat([e_dr, e_mass, torch.abs(Mass_i - Mass_j)], dim = 1)) 
        return edge, Pmc_j, edge_index[1]

    def aggregate(self, message, index, Pmc):
        edge_sc, cart, inc = message
        edge = edge_sc.max(dim = 1)[1]
        edge_sc = torch.nn.Softmax(dim = 1)(edge_sc)

        Pmc_i_sum = torch.zeros(Pmc.shape, dtype = torch.float, device = Pmc.device)
        Pmc_i_sum.index_add_(0, index[edge == 1], cart[edge == 1])
        mass = MassFromPxPyPzE(Pmc_i_sum)/1000
        return mass, edge, edge_sc
