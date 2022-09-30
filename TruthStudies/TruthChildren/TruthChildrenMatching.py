from AnalysisTopGNN.Generators import GenerateDataLoader
from AnalysisTopGNN.IO import UnpickleObject
from AnalysisTopGNN.Events import EventGraphTruthTopChildren
from Templates.EventFeatureTemplate import ApplyFeatures
from AnalysisTopGNN.Particles.Particles import Particle
from AnalysisTopGNN.Plotting import TH1F, CombineTH1F
from AnalysisTopGNN.Reconstruction import Reconstructor

import torch
from LorentzVector import *


def PlotTruthTopChildrenMass():
    ev = UnpickleObject("QU_0","./BasicBaseLineChildren/EventCache/BSM4Top/QU_0")
    
    ev_tops = {}
    for i in ev.Events:
        evnt = ev.Events[i]["nominal"]
        tops = {}
        for ch in evnt.TruthTopChildren:
            indx = ch.Index
            if indx not in tops:
                tops[indx] = Particle()
            tops[indx].Children.append(ch)
        
        ev_tops[i] = []
        for t in tops:
            tops[t].CalculateMass(tops[t].Children)
            ev_tops[i].append(tops[t].Mass_GeV)
       
    EV_D = [j for t in ev_tops.values() for j in t]
    Plot = {}
    Plot["Title"] = "EventGenerator"
    EG_H = TH1F(**Plot) 
    EG_H.xData = EV_D

    DL = GenerateDataLoader()
    DL.VerboseLevel = 0
    DL.chnk = 1000
    DL.EventGraph = EventGraphTruthTopChildren
    ApplyFeatures(DL, "TruthChildren")
    DL.AddSample(ev, "nominal", True, True)
    DL.ProcessSamples()
      
    Data = []
    Data_manual = []
    R = Reconstructor()
    for ev in DL.DataContainer:
        evnt = DL.DataContainer[ev]
        mass = R(evnt).MassFromEdgeFeature("edge")
        Data += mass.tolist()


        idx = evnt["E_T_edge"].view(-1)
        pt = evnt["N_pT"]
        eta = evnt["N_eta"]
        phi = evnt["N_phi"]
        e = evnt["N_energy"]
        ed_x = evnt["edge_index"]
        
        e_i, e_j = ed_x[0], ed_x[1]
        i_msk, j_msk = e_i[idx == 1], e_j[idx == 1]
        i_msk, j_msk = i_msk[i_msk != j_msk], j_msk[i_msk != j_msk] 
        
        Pmc = TensorToPxPyPzE(torch.cat([pt, eta, phi, e], dim = 1))
        Pmc_ = Pmc.clone()
        Pmc_.index_add_(0, i_msk, Pmc[j_msk])
        Pmc_ = torch.unique((Pmc_.to(dtype = torch.long)/1000).to(dtype = torch.long), dim = 0)
        Pmc_ = MassFromPxPyPzE(Pmc_)
        Data_manual += [k for t in Pmc_.tolist() for k in t]

    Plot["Title"] = "Reconstructor"
    DL_H = TH1F(**Plot) 
    DL_H.xData = Data

    Plot["Title"] = "Manual"
    DL_M = TH1F(**Plot) 
    DL_M.xData = Data_manual

    Plot["Title"] = "Top Mass From Truth Top Children (Post FSR) - Using Different Reconstructors"
    Plot["xTitle"] = "Invariant Mass (GeV)"
    Plot["yTitle"] = "Entries"
    Plot["DPI"] = 500
    Plot["Scaling"] = 1
    Plot["Filename"] = "TopsFromTruthTopChildren_FSR"
    Plot["xBins"] = 250
    Plot["xMin"] = 0
    Plot["xMax"] = 250
    Plot["Histograms"] = [EG_H, DL_M, DL_H]
   
    print(len(Data), len(Data_manual), len(EV_D))


    x = CombineTH1F(**Plot)
    x.SaveFigure("./Plots/TruthTopChildren/")
