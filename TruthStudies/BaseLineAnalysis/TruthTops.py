from AnalysisTopGNN.Plotting import TH1F, CombineTH1F

def TruthTops(Ana):    
    _leptons = [11, 12, 13, 14, 15, 16]
   
    TruthTopsRes = [] 
    TruthSpecTops = []
    ZPrimeMass = []

    ResonanceLepton = []
    ResonanceHadron = []
    SpectatorLepton = []
    SpectatorHadron = []
    
    NumTops = []
    lumi = 0
    for i in Ana:
        event = i.Trees["nominal"]
        lumi += event.weightmc 
       
        Zprime = []
        for t in event.TopPostFSR:
            if t.FromRes == 1:
                TruthTopsRes.append(t.CalculateMass())
                Zprime.append(t)
            else:
                TruthSpecTops.append(t.CalculateMass())
           
            _lep = False
            for c in t.Children:
                if abs(c.pdgid) in _leptons:
                    _lep = True
                    break
            
            if _lep and t.FromRes == 1:
                ResonanceLepton.append(t)

            elif _lep == False and t.FromRes == 1:
                ResonanceHadron.append(t)

            elif _lep and t.FromRes == 0:
                SpectatorLepton.append(t)

            elif _lep == False and t.FromRes == 0:
                SpectatorHadron.append(t)

        ZPrimeMass.append(sum(Zprime).CalculateMass())
        NumTops.append(len(event.TopPostFSR))

    Plots = {
                "Title" : "Invariant Mass of Truth Tops Originating\n from the Z' Resonance (1.5 TeV) (Final State Radiation)",
                "xTitle" : "Invariant Mass (GeV)",
                "yTitle" : "Entries (a.u.)",
                "xData" : TruthTopsRes, 
                "xMin" : 170, 
                "yMin" : 0, 
                "xMax" : 180,
                "xBins" : 100,
                "OutputDirectory" : "./Figures/TruthTops", 
                "Filename" : "Figure_1a",
                "Style" : "ATLAS",
                "ATLASLumi" : lumi,
            }
    T1a = TH1F(**Plots)
    T1a.SaveFigure() 

    Plots["Title"] = "Invariant Mass of Spectator Tops"
    Plots["Filename"] = "Figure_1c"
    Plots["xData"] = TruthSpecTops
    
    T1c = TH1F(**Plots)
    T1c.SaveFigure()

    Plots["Title"] = "Invariant Mass of Z' Resonance (1.5 TeV) derived from Truth Tops"
    Plots["Filename"] = "Figure_1b"
    Plots["xMin"] = 0
    Plots["xMax"] = None
    Plots["xData"] = ZPrimeMass

    T1b = TH1F(**Plots)
    T1b.SaveFigure()

    Plots["Title"] = "Number of Tops for Sampled Events"
    Plots["xTitle"] = "Number of Tops"
    Plots["xMin"] = 0
    Plots["xMax"] = None
    Plots["xData"] = NumTops
    Plots["xBins"] = max(NumTops) +1
    Plots["xBinCentering"] = True
    Plots["Filename"] = "Figure_1d"
    T1e = TH1F(**Plots)
    T1e.SaveFigure()


    Plots["Title"] = "Decay Modes of all Tops"
    Plots["xTickLabels"] = ["Res-Lep (" + str(len(ResonanceLepton)) + ")", 
                            "Res-Had (" + str(len(ResonanceHadron)) + ")", 
                            "Spec-Lep (" + str(len(SpectatorLepton)) + ")", 
                            "Spec-Had (" + str(len(SpectatorHadron)) + ")", 
                            "n-Top delta (" + str(abs(len(ResonanceLepton + ResonanceHadron) - len(SpectatorLepton + SpectatorHadron))) +")"]
    Plots["xTitle"] = "Decay Mode of Top (a.u.)"
    Plots["xData"] = [1, 2, 3, 4, 5]
    Plots["xWeights"] = [len(ResonanceLepton), 
                         len(ResonanceHadron), 
                         len(SpectatorLepton), 
                         len(SpectatorHadron), 
                         abs(len(ResonanceLepton + ResonanceHadron) - len(SpectatorLepton + SpectatorHadron))]
    Plots["xMin"] = 1
    Plots["xMax"] = 5
    Plots["xBins"] = 5
    Plots["xBinCentering"] = True
    Plots["Filename"] = "Figure_1e"
    
    T1d = TH1F(**Plots)
    T1d.SaveFigure()

    
    Plots = {
                "Title" : None, 
                "xTitle" : "Transverse Momenta (GeV)", 
                "yTitle" : "Entries",
                "xBins" : 200,
                "Style" : "ATLAS",
                "ATLASLumi" : lumi,
            }
    
    Plots["Title"] = "Res-Lep"
    Plots["xData"] = [t.pt/1000 for t in ResonanceLepton]
    RL = TH1F(**Plots)

    Plots["Title"] = "Res-Had"
    Plots["xData"] = [t.pt/1000 for t in ResonanceHadron]
    RH = TH1F(**Plots)

    Plots["Title"] = "Spec-Lep"
    Plots["xData"] = [t.pt/1000 for t in SpectatorLepton]
    SL = TH1F(**Plots)

    Plots["Title"] = "Spec-Had"
    Plots["xData"] = [t.pt/1000 for t in SpectatorHadron]
    SH = TH1F(**Plots)

    Plots["Title"] = "Transverse Momenta Distribution of Tops for Different Top Decay Modes"
    Plots["xData"] = []
    Plots["OutputDirectory"] = "./Figures/TruthTops"
    Plots["Histograms"] = [RL, RH, SL, SH]
    Plots["Filename"] = "Figure_1f"
    T1f = CombineTH1F(**Plots)
    T1f.SaveFigure()

