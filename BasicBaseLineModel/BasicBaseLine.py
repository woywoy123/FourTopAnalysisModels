import torch 
from torch_geometric.nn import MessagePassing
from torch.nn import Sequential as Seq, Linear, ReLU, Sigmoid, Tanh
from LorentzVector import *
from torch_geometric.utils import *

def MakeMLP(lay):
    out = []
    for i in range(len(lay)-1):
        x1, x2 = lay[i], lay[i+1]
        out += [Linear(x1, x2)]
    return Seq(*out)

class BasicBaseLineTruthJet(MessagePassing):

    def __init__(self):
        super().__init__(aggr = None, flow = "target_to_source")
        
        def MakeMLP(lay):
            out = []
            for i in range(len(lay)-1):
                x1, x2 = lay[i], lay[i+1]
                out += [Linear(x1, x2)]
            return Seq(*out)

        self.O_edge = None
        self.L_edge = "CEL"
        self.C_edge = True

        self.O_from_res = None
        self.L_from_res = "CEL"
        self.C_from_res = True

        self.O_signal_sample = None
        self.L_signal_sample = "CEL"
        self.C_signal_sample = True

        self.O_from_top = None
        self.L_from_top = "CEL"
        self.C_from_top = True
        
        end = 2048

        self._Node = MakeMLP([7, 256, 1024, end])
        self._Edge = MakeMLP([2, 256, 1024, end])

        self._isedge = Seq(Linear(2*end, int(end/2), False), ReLU(), Linear(int(end/2), 2, False))
        self._istop = Seq(Linear(2*end, int(end/2), False), ReLU(), Linear(int(end/2), 2, False))
        self._ResSw = Seq(Linear(end, int(end/2), False), ReLU(), Linear(int(end/2), 2, False))
        self._fromRes = Seq(Linear(end*3, int(end/2), False), ReLU(), Linear(int(end/2), 2, False))

        self._mass = MakeMLP([1, 1024, 1024, end])
        self._node_m = MakeMLP([4*end, 1024, end])

        self._signal = MakeMLP([8, 256, 256, 256, 2])
    
    def forward(self, i, edge_index, N_eta, N_energy, N_pT, N_phi, N_mass, N_islep, N_charge, G_mu, G_met, G_met_phi, G_pileup, G_njets, G_nlep):
        device = N_eta.device
        Pmu = torch.cat([N_pT, N_eta, N_phi, N_energy], dim = 1)
        Pmc = TensorToPxPyPzE(Pmu)
        Mass = N_mass/1000

        # Make Prediction about topology
        node_enc = self._Node(torch.cat([Pmc, Mass, N_islep, N_charge], dim = 1))
        node = self.propagate(edge_index, Pmu = Pmu, Pmc = Pmc, Mass = Mass, charge = N_charge, islep = N_islep, node_enc = node_enc)
        rw = self._ResSw(node_enc)
        res = rw.max(1)[1] 
        
        Mass_sum = torch.zeros((Pmu.shape[0], node.shape[1]), device = Pmu.device, dtype = torch.float)
        Mass_sum[res == 1] = self._mass(MassFromPxPyPzE(Pmc[res == 1].sum(0))/1000)
        self.O_from_res = self._fromRes(torch.cat([Mass_sum, node_enc, node], dim = 1)) + rw

        # Aggregate the nodes into a per graph basis if batches are more than 1.
        graph_res = self.O_from_res.view(i.shape[0], -1, 2).sum(dim = 1)
        self.O_signal_sample = self._signal(torch.cat([graph_res, G_mu, G_met, G_met_phi, G_pileup, G_njets, G_nlep], dim = 1))

        return self.O_edge, self.O_from_res
    
    def message(self, edge_index, Pmu_i, Pmu_j, Pmc_i, Pmc_j, Mass_i, Mass_j, charge_i, charge_j, islep_i, islep_j, node_enc_i, node_enc_j):
        dR = TensorDeltaR(Pmu_i, Pmu_j)

        mass = self._mass(MassFromPxPyPzE(Pmc_i + Pmc_j)/1000)
        edge = self._Edge(torch.cat([dR, islep_i*islep_j], dim = 1))
        self.O_edge = self._isedge(torch.cat([edge, mass], dim = 1))

        return self.O_edge, Pmc_j, node_enc_j, edge+mass

    def aggregate(self, message, index, Pmu, node_enc):
        edge, Pm_j, node_j, edgemlp = message
        sw = edge.max(dim = 1)[1]

        Mass_sum = torch.zeros(Pmu.shape, device = Pmu.device, dtype = torch.float)
        Mass_sum.index_add_(0, index[sw == 1], Pm_j[sw == 1])
        Mass = MassFromPxPyPzE(Mass_sum) / 1000 
        MassMLP = self._mass(Mass)
        self.O_from_top = self._istop(torch.cat([MassMLP, node_enc], dim = 1))

        Node_sum = torch.zeros((Pmu.shape[0], node_j.shape[1]), device = Pmu.device, dtype = torch.float)
        Node_sum.index_add_(0, index[sw == 1], node_j[sw == 1])

        edge_sum = torch.zeros((Pmu.shape[0], edgemlp.shape[1]), device = Pmu.device, dtype = torch.float)
        edge_sum.index_add_(0, index[sw == 1], edgemlp[sw == 1])

        return self._node_m(torch.cat([MassMLP, node_enc, Node_sum, edge_sum], dim = 1))


