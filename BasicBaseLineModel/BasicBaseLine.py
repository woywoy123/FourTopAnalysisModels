import torch 
from torch_geometric.nn import MessagePassing
from torch.nn import Sequential as Seq, Linear, ReLU, Sigmoid, Tanh
import torch.nn.functional as F
from LorentzVector import *
from torch_geometric.utils import to_dense_adj, add_remaining_self_loops, dense_to_sparse

torch.set_printoptions(4, profile = "full", linewidth = 100000)


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

        end = 1024
        self._isDR = Seq(Linear(2, end), Sigmoid(), Linear(end, end), Linear(end, end), Sigmoid(), Linear(end, 2))
        self._isMass = Seq(Linear(2, end), Sigmoid(), Linear(end, end), Linear(end, end), Sigmoid(), Linear(end, 2))
        self._topo = Seq(Linear(6, end), Sigmoid(), Linear(end, end), Sigmoid(), Linear(end, end), Linear(end, 2))
        self._it = 0

    def forward(self, i, edge_index, E_T_edge, N_pT, N_eta, N_phi, N_energy, N_mass):
        
        Pmu = torch.cat([N_pT, N_eta, N_phi, N_energy], dim = 1)
        Pmc = TensorToPxPyPzE(Pmu)
        
        self.Counter = torch.zeros(to_dense_adj(edge_index)[0].shape, device = Pmu.device)

        edge_index_new1 = self.propagate(edge_index, Pmc = Pmc, Pmu = Pmu, Mass = N_mass, T_edge = E_T_edge)

        return self.O_edge

    def message(self, edge_index, Pmc_i, Pmc_j, Pmu_i, Pmu_j, Mass_i, Mass_j, T_edge):
        e_dr = TensorDeltaR(Pmu_i, Pmu_j)
        e_mass = MassFromPxPyPzE(Pmc_i + Pmc_j) / 1000
        dR = self._isDR(torch.cat([e_dr, torch.abs(Mass_i - Mass_j)], dim = 1)) 
        mass = self._isMass(torch.cat([e_mass, torch.abs(Mass_i - Mass_j)], dim = 1)) 
        return dR+mass, edge_index, e_mass, mass, dR

    def aggregate(self, message, index, Pmc, Pmu, Mass):
        edge_sc, edge_index, e_mass, mlp_mass, dR = message
        edge = edge_sc.max(dim = 1)[1]

        self.Counter[edge_index[0], edge_index[1]] = edge.view(-1).type(torch.float) 
        
        Pmc_sum = torch.zeros(Pmc.shape, device = Pmc.device)
        Pmc_sum.index_add_(0, edge_index[0][edge == 1], Pmc[edge_index[1][edge == 1]])
        mass = MassFromPxPyPzE(Pmc_sum)/1000
       
        self.Counter = self.Counter*mass
        mass_v = self.Counter[edge_index[0], edge_index[1]].view(-1, 1)
        topo = self._topo(torch.cat([mass_v, dR, mlp_mass, torch.abs(mass_v - e_mass)], dim = 1))
        topo = F.dropout(topo, training = self.training)
        self.O_edge = topo
        self.Counter[self.Counter >= 1] = 1

        return dense_to_sparse(self.Counter)[0]




class BasicBaseLineRecursion(MessagePassing):

    def __init__(self):
        super().__init__(aggr = None, flow = "target_to_source")
        self.O_edge = None
        self.L_edge = "CEL"
        self.C_edge = True

        end = 1024
        self._isMass = Seq(Linear(1, end), Sigmoid(), Linear(end, end), Sigmoid(), Linear(end, 2))
        self._topo = Seq(Linear(6, end), Sigmoid(), Linear(end, end), Sigmoid(), Linear(end, 2))
        self._it = 0

    def forward(self, i, edge_index, E_T_edge, N_pT, N_eta, N_phi, N_energy, N_mass):
        if self._it == 0:
            self.device = N_pT.device
            self.edge_mlp = torch.zeros((edge_index.shape[1], 2), device = self.device)
            self.edge_count = torch.ones((edge_index.shape[1], 1), device = self.device)

        Pmu = torch.cat([N_pT, N_eta, N_phi, N_energy], dim = 1)
        Pmc = TensorToPxPyPzE(Pmu)
        
        edge_index = self.propagate(edge_index, Pmc = Pmc, Pmu = Pmu, Mass = N_mass, T_edge = E_T_edge)

        exit()
        return self.O_edge

    def message(self, edge_index, Pmc_i, Pmc_j, Pmu_i, Pmu_j, Mass_i, Mass_j, T_edge):
        e_dr = TensorDeltaR(Pmu_i, Pmu_j)
        e_mass = MassFromPxPyPzE(Pmc_i + Pmc_j) / 1000
        mlp_mass = self._isMass(e_mass) #torch.cat([e_mass, torch.abs(Mass_i - Mass_j)], dim = 1)) 
        return edge_index[1], mlp_mass, e_mass, Pmc_j

    def aggregate(self, message, index, Pmc, Pmu, Mass):
        edge_index, mlp_mass, e_mass, Pmc_j = message
        edge = mlp_mass.max(dim = 1)[1]

        self.edge_count[index == edge_index] = 0
        Pmc_i = Pmc_j[index == edge_index] # Self loops
        Pmc_i.index_add_(0, index[edge == 1], (Pmc_j*self.edge_count)[edge == 1])

        self.edge_mlp += mlp_mass
        self.edge_count[edge == 1] = 0

        return TensorToPtEtaPhiE(Pmc_i), MassFromPxPyPzE(Pmc_i)

