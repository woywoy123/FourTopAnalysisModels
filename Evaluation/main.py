from AnalysisTopGNN.Tools.Metrics import Metrics 
from AnalysisTopGNN.IO import Directories, WriteDirectory
from AnalysisTopGNN.IO import UnpickleObject, PickleObject
from AnalysisTopGNN.Plotting import *

from HelperFunctions import *
from ModelFunctions import *


SourceDir = "../BasicBaseLineModel/Dump/" # /CERN/AnalysisModelTraining/BasicBaseLineTruthJetSmall"
TargetDir = "./BasicBaseLineModel/"

D = Directories(SourceDir + "/Models")
D.ListDirs()
Dir = [i.split("/")[-1] for i in D.Files]
out = GetSampleDetails(SourceDir)
#BuildSymlinksToHDF5(SourceDir)
#Data = RetrieveSamples(SourceDir)
#PickleObject(Data, "HDF5")
Data = UnpickleObject("HDF5")

mp = {}
for i in out["DataContainer"]:
    hash_ = out["DataContainer"][i]
    if hash_ in Data:
        out["DataContainer"][i] = Data[hash_]
        mp[i] = hash_
    else:
        out["DataContainer"][i] = None

# Get a statistical breakdown of the data:
# -> All; 
#   n-Nodes, entries for process, Training/Validation of nodes

DC = out["DataContainer"]
Tr = out["TrainingSample"]
Val = out["ValidationSample"]

# Reverse the lookup to hash = node
Tr = [i for j in Tr for i in Tr[j]]
Val = [i for j in Val for i in Val[j]]

## Node Statistics - General Sample
#NodeStatistics(TargetDir, DC, mp, Tr, Val)
#
## Process Statistics
#ProcessStatistics(TargetDir, DC, mp, Tr, Val, out)


Name = [
    {"name":"edge", "node":"%252", "loss":"CEL", "classifier" : True},
    {"name":"from_res", "node":"556"},
    {"name":"signal_sample", "node":"573"}, 
    {"name":"from_top", "node":"347"}
    ]

M = ModelComparison(Data)
for i in Dir:
    if "_" not in i:
        continue
    print(i)
    Ev = Evaluation(SourceDir + "/Models/", i)
    Ev.ReadStatistics()
    Ev.EpochLoop()
    #Ev.MakePlots(TargetDir)
    Ev.MakeLog(TargetDir)
    M.AddModel(Ev, Name)
M.RebuildMassEdge("edge")
M.RebuildMassNode("from_res")
M.MakePlots()



#TS = TorchScriptModel("./Epoch_0_100.pt", inpt = Name)
##TS.AddOutput("edge", "252", "CEL", True)
##TS.AddOutput("from_res", "556")
##TS.AddOutput("signal_sample", "573")
##TS.AddOutput("from_top", "347")
#TS.Finalize()
#TS.to("cuda")
#
##PickleObject(Data, "tmp")
#Data = UnpickleObject("tmp")
#r = Reconstructor(TS)
#r.TruthMode = False
#for i in Data:
#    smpl = Data[i]
#    smpl.to("cuda") 
#    TS(**smpl.to_dict())
#    mass = r(smpl).MassFromNodeFeature("from_top")
#    print(mass) 
#    mass = r(smpl).MassFromEdgeFeature("edge")
#    print(mass)





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

