# Once written, put this code into the main repository!
from AnalysisTopGNN.IO import UnpickleObject, ExportToDataScience, Directories
from AnalysisTopGNN.Tools import Threading
import os
import random
from AnalysisTopGNN.Plotting import *

def GetSampleDetails(Directory):
    Dir = Directory + "/DataLoaderFileTraceDump/"
    DataContainer = Dir + "/DataContainer.pkl"
    FileTrace = Dir + "/FileTraces.pkl"
    TrainingSample = Dir + "/TrainingSample.pkl"
    ValidationSample = Dir + "/ValidationSample.pkl"

    DataContainer = UnpickleObject(DataContainer) # <= holds path Hash for each event 
    FileTrace = UnpickleObject(FileTrace) # <= Holds the index for each file 
    TrainingSample = UnpickleObject(TrainingSample) # <= Holds the hash for each n-nodes
    ValidationSample = UnpickleObject(ValidationSample)

    return {"DataContainer" : DataContainer, 
            "FileTrace" : FileTrace, 
            "TrainingSample" : TrainingSample, 
            "ValidationSample" : ValidationSample}

def BuildSymlinksToHDF5(Directory):
    D = Directories()
    D.VerboseLevel = 0
    x = D.ListFilesInDir(Directory + "/DataLoader")
    dest = Directory + "/DataLoader/HDF5/"
    
    for i in x:
        name = i.split("/")[-1]
        os.symlink(i, dest + name)
        print(name)

def RetrieveSamples(Directory):
    #def function(F, inpt = Directory + "/DataLoader/HDF5/", out = []):
    #    exp = ExportToDataScience()
    #    exp.VerboseLevel = 0
    #    for i in F:
    #        out.append([i, exp.ImportEventGraph(i, inpt)])
    #    del exp
    #    return out

   
    #TH = Threading(list(F.values()), function, threads = 6)
    #TH.Start()
    #Data = {}
    #for i in TH._lists:
    #    Data[i[0]] = i[1] 

    F = UnpickleObject(Directory + "/DataLoaderFileTraceDump/DataContainer.pkl")
    Data = {}
    exp = ExportToDataScience()
    
    lst = list(F)
    random.shuffle(lst)

    it = 0
    for i in lst:
        print("--> " + str(round(float(it / 1000)*100, 4)) + "%")
        Data[F[i]] = list(exp.ImportEventGraph(F[i], Directory + "/DataLoader/HDF5/").values())[0]
        it+=1
        
        if it == 1000:
            break
    return Data

def FileLookup(Dict, i):

    File = Dict["FileTrace"]
    Start = File["Start"]
    End = File["End"]
    Samples = File["Samples"]
    
    it = 0
    for s, e in zip(Start, End):
        if s <= i and i <= e:
            break
        it +=1 
    return Samples[it].split("/")[-2]


def NodeStatistics(TargetDir, DataContainer, HashMap, TrainingSample, ValidationSample):

    all_, val_, tr_ = [], [], []
    for i in DataContainer:
        if DataContainer[i] == None:
            continue
        obj = DataContainer[i]
        hsh = HashMap[i]
    
        n_nd = int(obj.num_nodes)
        if hsh in TrainingSample:
            tr_.append(n_nd)
        if hsh in ValidationSample:
            val_.append(n_nd)
        all_.append(n_nd)
    
    
    SampleDist = TH1F()
    SampleDist.Alpha = 0.5
    SampleDist.Title = "All"
    SampleDist.xData = all_
    SampleDist.Normalize = False
    SampleDist.xBinCentering = True
    SampleDist.Filename = "n-Nodes_All"
    SampleDist.SaveFigure(TargetDir + "SampleStatistics/Raw")
    
    ValDist = TH1F()
    ValDist.Alpha = 0.5
    ValDist.Title = "Validation"
    ValDist.xData = val_
    ValDist.Normalize = False
    ValDist.xBinCentering = True
    ValDist.Filename = "n-Nodes_Validation"
    ValDist.SaveFigure(TargetDir + "SampleStatistics/Raw")
    
    TrainDist = TH1F()
    TrainDist.Alpha = 0.5
    TrainDist.Title = "Training"
    TrainDist.xData = tr_
    TrainDist.Normalize = False
    TrainDist.xBinCentering = True
    TrainDist.Filename = "n-Nodes_Training"
    TrainDist.SaveFigure(TargetDir + "SampleStatistics/Raw")
    
    Merged = CombineTH1F()
    Merged.Title = "Node Distribution for Training/Validation Data Superimposed\n over Complete Sample"
    Merged.Stack = False
    Merged.xBinCentering = True
    Merged.Style = "ATLAS"
    Merged.Normalize = "%"
    Merged.Histogram = SampleDist
    Merged.Histograms = [ValDist, TrainDist]
    Merged.xTitle = "Number of Nodes in Graph"
    Merged.yTitle = "Percentage of n-Nodes (%)"
    Merged.Filename = "n-Nodes_Combined"
    Merged.SaveFigure(TargetDir + "SampleStatistics")

def ProcessStatistics(TargetDir, DataContainer, HashMap, TrainingSample, ValidationSample, All):
    all_, val_, tr_ = {}, {}, {}
    all_n, val_n, tr_n = [], [], []
    for i in DataContainer:
        if DataContainer[i] == None:
            continue
        obj = DataContainer[i]
        hsh = HashMap[i]
    
        n_nd = int(obj.num_nodes)
        indx = int(obj.i) 
        
        smpl_ = FileLookup(All, indx)
        if smpl_ not in all_:
            all_[smpl_] = []
            val_[smpl_] = []
            tr_[smpl_] = []
        
        if hsh in TrainingSample:
            tr_[smpl_].append(n_nd)
            tr_n.append(n_nd)

        if hsh in ValidationSample:
            val_[smpl_].append(n_nd)
            val_n.append(n_nd)

        all_[smpl_].append(n_nd)
        all_n.append(n_nd)
   

    All = TH1F()
    All.Title = "All"
    All.xData = all_n
    All.xBinCentering = True
    All.Alpha = 0.5

    Merged = CombineTH1F()
    Merged.Title = "Node Distribution of Processes Superimposed\n over Complete Sample"
    Merged.Normalize = "%"
    Merged.Histogram = All
    Merged.xBinCentering = True
    Merged.Style = "ATLAS"

    for i in all_:
        Hist = TH1F()
        
        Hist.Title = i
        Hist.xData = all_[i]
        Hist.Alpha = 0.5
        Hist.xBinCentering = True
        
        Hist.Filename = "all_" + i
        Hist.SaveFigure(TargetDir + "SampleStatistics/Raw")
        Merged.Histograms.append(Hist)
    
    Merged.xTitle = "Number of Nodes in Graph"
    Merged.yTitle = "Percentage of n-Nodes (%)"
    Merged.Filename = "ProcessDistribution_All"
    Merged.SaveFigure(TargetDir + "SampleStatistics")
