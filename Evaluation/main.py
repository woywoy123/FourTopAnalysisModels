from AnalysisTopGNN.Tools.Metrics import Metrics 
from AnalysisTopGNN.IO import Directories, WriteDirectory
from AnalysisTopGNN.IO import UnpickleObject, PickleObject
from AnalysisTopGNN.Plotting import *


from HelperFunctions import *
from ModelFunctions import *


SourceDir = "/CERN/AnalysisModelTraining/BasicBaseLineTruthJetSmall"
TargetDir = "./BasicBaseLineModel/"

D = Directories(SourceDir)
D.ListDirs()
Dir = [i.split("/")[-1] for i in D.Files]

#out = GetSampleDetails(SourceDir)
#BuildSymlinksToHDF5(SourceDir)
#Data = RetrieveSamples(SourceDir)
#PickleObject(Data, "HDF5")
Data = UnpickleObject("HDF5")
#
#mp = {}
#for i in out["DataContainer"]:
#    hash_ = out["DataContainer"][i]
#    if hash_ in Data:
#        out["DataContainer"][i] = Data[hash_]
#        mp[i] = hash_
#    else:
#        out["DataContainer"][i] = None
#
## Get a statistical breakdown of the data:
## -> All; 
##   n-Nodes, entries for process, Training/Validation of nodes
##
#DC = out["DataContainer"]
#Tr = out["TrainingSample"]
#Val = out["ValidationSample"]
#
## Reverse the lookup to hash = node
#Tr = {i : j for j in Tr for i in Tr[j]} 
#Val = {i : j for j in Val for i in Val[j]}
#
#
## Node Statistics - General Sample
#NodeStatistics(TargetDir, DC, mp, Tr, Val)
#
## Process Statistics
#ProcessStatistics(TargetDir, DC, mp, Tr, Val, out)
#



M = ModelComparison()
pro = [
        {"name": "edge", "node" : "%234", "classification" : "True", "loss" : "CEL"}, 
        {"name": "from_res", "node" : "%512", "classification" : "True", "loss" : "CEL"}, 
        #{"name": "signal_sample", "node" : "%538", "classification" : "True", "loss" : "CEL"}, 
        #{"name": "from_top", "node" : "%329", "classification" : "True", "loss" : "CEL"}, 
    ]
M.Device = "cuda"
#W = WriteDirectory()
#for i in Dir:
#    if "_" not in i:
#        continue
#
#    #W.MakeDir(TargetDir + i)
#    #M = Metrics(i, SourceDir)
#    #M.PlotStats(TargetDir + i)
#
#    Ev = Evaluation(SourceDir, i)
#    Ev.ReadStatistics()
#    Ev.EpochLoop()
#    #Ev.MakePlots(TargetDir)
#    Ev.MakeLog(TargetDir)
#    break
#PickleObject(Ev, "Model")
Ev = UnpickleObject("Model")
M.AddModel(Ev, pro)
#M.RebuildMassEdge(Data, "pT", "eta", "phi", "energy", "edge")
#M.RebuildMassNode(Data, "pT", "eta", "phi", "energy", "from_res")
M.MakePlots()































##Data = list(Data.values())[0]
###model = BasicBaseLineTruthJet
###m = torch.jit.trace(model(), inpt_list)
###torch.jit.save(m, "./Model.pt")
##m = torch.jit.load("./Model.pt")
##
##inpt_keys = [k.replace(")", "").replace(",", "") for k in str(m.forward.schema).split(" ") if "Tensor" not in k and "->" not in k and "self" not in k][1:]
##inpt_keys = {k : Data[k] for k in inpt_keys}
##
##
#print(dir(m))
#print(dir(m.graph))
#
#get = []
#for i in list(m.graph.nodes()):
#    #if str(i).startswith("%234"):
#    #    m.graph.registerOutput(list(i.outputs())[0])
#
#    ###print(str(i).replace("\n", ""))
#    #if str(i).startswith("%_signal"):
#    #    m.graph.registerOutput(list(i.outputs())[0])
#    #
#    #if str(i).startswith("%329"):
#    #    m.graph.registerOutput(list(i.outputs())[0])
#
#    if str(i).startswith("%input8.1") or str(i).startswith("%_signal.1"):
#        m.graph.registerOutput(list(i.outputs())[0])
#        get.append(i)
#
#for i in get:
#    print(i)
#
##for i in list(m.graph.nodes()):
##    print(str(i).replace("\n", ""))
#
#
#
#
#
##
##
##
#m.graph.makeMultiOutputIntoTuple()
#m.eval()
#out = m(**inpt_keys)
##
#
#
#
#for i in out[1:]:
#    print("_--")
#    if isinstance(i, torch._C.ScriptModule):
#        print(out[-1])
#        print(m._isedge(out[-1]))
#    exit()
#    print(i)
##exit()
##
##
##out = [i for i in out][1:]
##print(out)
#
##print(list(m.graph.outputs()))
##print(m.graph.return_node())

