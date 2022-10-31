from AnalysisTopGNN.Plotting import TH1F, CombineTH1F

def TruthChildren(Ana):
    _leptons = [11, 12, 13, 14, 15, 16]

    NChildrenResTopLep = []
    NChildrenResTopHad = []
    
    NChildrenSpecTopLep = []
    NChildrenSpecTopHad = []

    lumi = 0 
    for i in Ana:
        event = i.Trees["nominal"]
        lumi += event.weightmc
        tops = event.TopPostFSR
        
        for t in tops:
            _lep = False
            
            if len([c for c in t.Children if abs(c.pdgid) in _leptons]) > 0:
                _lep = True
            
            if t.FromRes == 1 and _lep:
                NChildrenResTopLep.append(t)

            elif t.FromRes == 1 and _lep == False:
                NChildrenResTopHad.append(t)

            elif t.FromRes == 0 and _lep:
                NChildrenSpecTopLep.append(t)

            elif t.FromRes == 0 and _lep == False:
                NChildrenSpecTopHad.append(t) 

    Plots = {
                "Title" : "Number of Decay Products from Tops",
                "xTitle" : "Number of Children",
                "yTitle" : "Entries (a.u.)",
                "xData" : [], 
                "xMin" : 0, 
                "yMin" : 0, 
                "xMax" : None,
                "xBins" : None,
                "OutputDirectory" : "./Figures/TopChildren", 
                "Style" : "ATLAS",
                "ATLASLumi" : lumi,
            }

    Plots["Title"] = "Res-Lep"
    Plots["xData"] = [len(t.Children) for t in NChildrenResTopLep]
    RL = TH1F(**Plots)

    Plots["Title"] = "Res-Had"
    Plots["xData"] = [len(t.Children) for t in NChildrenResTopHad]
    RH = TH1F(**Plots)

    Plots["Title"] = "Spec-Lep"
    Plots["xData"] = [len(t.Children) for t in NChildrenSpecTopLep]
    SL = TH1F(**Plots)

    Plots["Title"] = "Spec-Had"
    Plots["xData"] = [len(t.Children) for t in NChildrenSpecTopHad]
    SH = TH1F(**Plots)

    Plots["Title"] = "Number of Decay Products from Tops"
    Plots["xData"] = []
    Plots["Histograms"] = [RL, RH, SL, SH]
    Plots["Stack"] = True
    Plots["xBinCentering"] = True
    Plots["xStep"] = 1
    Plots["Filename"] = "Figure_1a"
    T1a = CombineTH1F(**Plots)
    T1a.SaveFigure()
   


    Plots = {
                "xTitle" : "Transverse Momenta of Child (GeV)",
                "yTitle" : "Entries (a.u.)",
                "xData" : [], 
                "xMin" : 0, 
                "yMin" : 0, 
                "xMax" : None,
                "xBins" : None,
                "OutputDirectory" : "./Figures/TopChildren", 
                "Style" : "ATLAS",
                "xMax" : 750,
                "ATLASLumi" : lumi,
            }

    Plots["Title"] = "Res-Lep"
    Plots["xData"] = [c.pt/1000 for t in NChildrenResTopLep for c in t.Children]
    RL = TH1F(**Plots)

    Plots["Title"] = "Res-Had"
    Plots["xData"] = [c.pt/1000 for t in NChildrenResTopHad for c in t.Children]
    RH = TH1F(**Plots)

    Plots["Title"] = "Spec-Lep"
    Plots["xData"] = [c.pt/1000 for t in NChildrenSpecTopLep for c in t.Children]
    SL = TH1F(**Plots)

    Plots["Title"] = "Spec-Had"
    Plots["xData"] = [c.pt/1000 for t in NChildrenSpecTopHad for c in t.Children]
    SH = TH1F(**Plots)

    Plots["Title"] = "Transverse Momenta Distribution of Top Decay Products \n under Different Decay Modes"
    Plots["xData"] = []
    Plots["Histograms"] = [RL, RH, SL, SH]
    Plots["Filename"] = "Figure_1b"
    Plots["Stack"] = True
    Plots["xStep"] = 50
    T1b = CombineTH1F(**Plots)
    T1b.SaveFigure()
 


    Plots = {
                "xTitle" : "Invariant Mass (GeV)",
                "yTitle" : "Entries (a.u.)",
                "xData" : [], 
                "xMin" : 100, 
                "yMin" : 0, 
                "xMax" : 200,
                "xBins" : None,
                "OutputDirectory" : "./Figures/TopChildren", 
                "Style" : "ATLAS",
                "ATLASLumi" : lumi,
            }

    Plots["Title"] = "Res-Lep"
    Plots["xData"] = [sum(t.Children).CalculateMass() for t in NChildrenResTopLep if len(t.Children) != 0]
    RL = TH1F(**Plots)

    Plots["Title"] = "Res-Had"
    Plots["xData"] = [sum(t.Children).CalculateMass() for t in NChildrenResTopHad if len(t.Children) != 0]
    RH = TH1F(**Plots)

    Plots["Title"] = "Spec-Lep"
    Plots["xData"] = [sum(t.Children).CalculateMass() for t in NChildrenSpecTopLep if len(t.Children) != 0]
    SL = TH1F(**Plots)

    Plots["Title"] = "Spec-Had"
    Plots["xData"] = [sum(t.Children).CalculateMass() for t in NChildrenSpecTopHad if len(t.Children) != 0]
    SH = TH1F(**Plots)

    Plots["Title"] = "Invariant Mass of Reconstructed Top Quark from Decay Products \n under Different Decay Modes"
    Plots["xData"] = []
    Plots["Histograms"] = [RL, RH, SL, SH]
    Plots["Filename"] = "Figure_1c"
    Plots["Stack"] = False
    Plots["xStep"] = 5
    T1c = CombineTH1F(**Plots)
    T1c.SaveFigure()
 


    Plots = {
                "xTitle" : "$\Delta$R Between Children (a.u.)",
                "yTitle" : "Entries (a.u.)",
                "xData" : [], 
                "xMin" : 0, 
                "yMin" : 0, 
                "xMax" : None,
                "xBins" : None,
                "OutputDirectory" : "./Figures/TopChildren", 
                "Style" : "ATLAS",
                "ATLASLumi" : lumi,
            }

    def DeltaR(t):
        done = []
        out = []
        for c in t.Children:
            for x in t.Children:
                if x == c:
                    continue
                if x in done:
                    continue
                out.append(c.DeltaR(x))
            
            done.append(c)
        return out
    
    Plots["Title"] = "Res-Lep"
    Plots["xData"] = [dr for t in NChildrenResTopLep for dr in DeltaR(t)]
    RL = TH1F(**Plots)

    Plots["Title"] = "Res-Had"
    Plots["xData"] = [dr for t in NChildrenResTopHad for dr in DeltaR(t)]
    RH = TH1F(**Plots)

    Plots["Title"] = "Spec-Lep"
    Plots["xData"] = [dr for t in NChildrenSpecTopLep for dr in DeltaR(t)]
    SL = TH1F(**Plots)

    Plots["Title"] = "Spec-Had"
    Plots["xData"] = [dr for t in NChildrenSpecTopHad for dr in DeltaR(t)]
    SH = TH1F(**Plots)

    Plots["Title"] = "$\Delta$R between Children of Common Top \n under Different Decay Modes"
    Plots["xData"] = []
    Plots["Histograms"] = [RL, RH, SL, SH]
    Plots["Filename"] = "Figure_1d"
    Plots["xStep"] = 0.4
    Plots["Stack"] = True
    T1d = CombineTH1F(**Plots)
    T1d.SaveFigure()
 

    def DeltaRTop(t):
        out = []
        for c in t.Children:
            out.append(t.DeltaR(c))
        return out



    Plots = {
                "xTitle" : "$\Delta$R Between Children and Top (a.u.)",
                "yTitle" : "Entries (a.u.)",
                "xData" : [], 
                "xMin" : 0, 
                "yMin" : 0, 
                "xMax" : None,
                "xBins" : None,
                "OutputDirectory" : "./Figures/TopChildren", 
                "Style" : "ATLAS",
                "ATLASLumi" : lumi,
            }

    Plots["Title"] = "Res-Lep"
    Plots["xData"] = [dr for t in NChildrenResTopLep for dr in DeltaRTop(t)]
    RL = TH1F(**Plots)

    Plots["Title"] = "Res-Had"
    Plots["xData"] = [dr for t in NChildrenResTopHad for dr in DeltaRTop(t)]
    RH = TH1F(**Plots)

    Plots["Title"] = "Spec-Lep"
    Plots["xData"] = [dr for t in NChildrenSpecTopLep for dr in DeltaRTop(t)]
    SL = TH1F(**Plots)

    Plots["Title"] = "Spec-Had"
    Plots["xData"] = [dr for t in NChildrenSpecTopHad for dr in DeltaRTop(t)]
    SH = TH1F(**Plots)

    Plots["Title"] = "$\Delta$R between Children and Top of Origin \n under Different Decay Modes"
    Plots["xData"] = []
    Plots["Histograms"] = [RL, RH, SL, SH]
    Plots["Filename"] = "Figure_1e"
    Plots["xStep"] = 0.4
    Plots["Stack"] = True
    T1d = CombineTH1F(**Plots)
    T1d.SaveFigure()
 
