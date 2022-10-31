from AnalysisTopGNN.Generators import Analysis
from AnalysisTopGNN.Events import Event
from AnalysisTopGNN.IO import UnpickleObject, PickleObject
from AnalysisTopGNN.Plotting import TH1F, CombineTH1F

from TruthTops import TruthTops
from TruthChildren import TruthChildren

def TruthJet(Ana):
   
    
    nLep = []
    nJets = []
    for i in Ana:
        event = i.Trees["nominal"]
        nLep.append(len(event.Leptons))
        nJets.append(len(event.Jets))





























if __name__ == '__main__':

    File = "/home/tnom6927/Downloads/CustomAnalysisTopOutputTest/tttt/QU_0.root"
    Ana = Analysis()
    Ana.ProjectName = "TruthTops"
    Ana.Event = Event
    Ana.EventCache = False
    Ana.DumpPickle = True
    Ana.InputSample("bsm4-top", File)
    Ana.Launch()

    #TruthTops(Ana)
    #TruthChildren(Ana)
