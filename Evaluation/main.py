from AnalysisTopGNN.Tools import Metrics 
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
##BuildSymlinksToHDF5(SourceDir)
##Data = RetrieveSamples(SourceDir)
##PickleObject(Data, "HDF5")
#Data = UnpickleObject("HDF5")
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
#
#DC = out["DataContainer"]
#Tr = out["TrainingSample"]
#Val = out["ValidationSample"]
#
## Reverse the lookup to hash = node
#Tr = {i : j for j in Tr for i in Tr[j]} 
#Val = {i : j for j in Val for i in Val[j]}


# Node Statistics - General Sample
#NodeStatistics(TargetDir, DC, mp, Tr, Val)

# Process Statistics
#ProcessStatistics(TargetDir, DC, mp, Tr, Val, out)


W = WriteDirectory()
for i in Dir:
    if "_" not in i:
        continue
    break

#W.MakeDir(TargetDir + i)
#M = Metrics(i, SourceDir)
#M.PlotStats(TargetDir + i)


Ev = Evaluation(SourceDir, i)
Ev.ReadStatistics()
Ev.EpochLoop()
#Ev.MakePlots(TargetDir)
Ev.MakeLog(TargetDir)

#ReadStatistics(SourceDir + "/" + i)

