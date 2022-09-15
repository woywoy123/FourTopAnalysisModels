# Once written, put this code into the main repository!
from AnalysisTopGNN.IO import UnpickleObject, ExportToDataScience, Directories
from AnalysisTopGNN.Tools import Threading
import os
import random
from AnalysisTopGNN.Plotting import *

def GetSampleDetails(Directory):
    Dir = Directory + "/FileTraces/"
    #DataContainer = Dir + "/DataContainer.pkl"
    FileTrace = Dir + "/FileTraces.pkl"
    TrainingSample = Dir + "/TrainingSample.pkl"
    #ValidationSample = Dir + "/ValidationSample.pkl"
    

    FileTrace = UnpickleObject(FileTrace) # <= Holds the index for each file 
    TrainingSample = UnpickleObject(TrainingSample) # <= Holds the hash for each n-nodes

    return {"DataContainer" : FileTrace["SampleMap"], 
            "FileTrace" : {k : FileTrace[k] for k in FileTrace if k != "SampleMap"}, 
            "TrainingSample" : TrainingSample["Training"], 
            "ValidationSample" : TrainingSample["Validation"]}

def BuildSymlinksToHDF5(Directory):
    D = Directories()
    D.VerboseLevel = 0
    x = D.ListFilesInDir(Directory + "/DataCache", [".hdf5"])
    dest = Directory + "/HDF5/"
    
    l = len(x)
    it = 0
    for i in x:
        name = i.split("/")[-1]
        os.symlink(i, dest + name)
        print(name, round(float(it/l)*100, 3))
        it+=1


def RetrieveSamples(Directory):


    F = UnpickleObject(Directory + "/FileTraces/FileTraces.pkl")
    F = F["SampleMap"]
    Data = {}
    exp = ExportToDataScience()
    
    lst = list(F)
    random.shuffle(lst)

    it = 0
    lim = 100 #len(lst)
    for i in lst:
        print("--> " + str(round(float(it / lim)*100, 4)) + "%")
        Data[F[i]] = list(exp.ImportEventGraph(F[i], Directory + "/HDF5/").values())[0]
        it+=1
        
        if it == lim:
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
        
        n_nd = int(obj.num_nodes)
        
        if i in TrainingSample:
            tr_.append(n_nd)
        if i in ValidationSample:
            val_.append(n_nd)
        all_.append(n_nd)
    
    
    SampleDist = TH1F()
    SampleDist.Alpha = 0.5
    SampleDist.Title = "All"
    SampleDist.xData = all_
    SampleDist.xBinCentering = True
    SampleDist.Filename = "n-Nodes_All"
    SampleDist.SaveFigure(TargetDir + "SampleStatistics/Raw")
    SampleDist.Color = None
    
    ValDist = TH1F()
    ValDist.Alpha = 0.5
    ValDist.Title = "Validation"
    ValDist.xData = val_
    ValDist.xBinCentering = True
    ValDist.Filename = "n-Nodes_Validation"
    ValDist.SaveFigure(TargetDir + "SampleStatistics/Raw")
    ValDist.Color = None 

    TrainDist = TH1F()
    TrainDist.Alpha = 0.5
    TrainDist.Title = "Training"
    TrainDist.xData = tr_
    TrainDist.xBinCentering = True
    TrainDist.Filename = "n-Nodes_Training"
    TrainDist.SaveFigure(TargetDir + "SampleStatistics/Raw")
    TrainDist.Color = None 

    Merged = CombineTH1F()
    Merged.Title = "Node Distribution for Training/Validation Data Superimposed\n over Complete Sample"
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
    def MakePlot(nodes_dict, nodes_list, Title1, Title2):
        H1 = TH1F(Title = Title1, xData = nodes_list, xBinCentering = True, Alpha = 0.5)
        
        H2 = CombineTH1F(Title = "Node Distribution of Processes Superimposed\n over " + Title2 + " Sample", 
                Normalize = "%", 
                Histogram = H1, 
                xBinCentering = True, 
                Style = "ATLAS", 
                xTitle = "Number of Nodes in Graph", 
                yTitle = "Percentage of n-Nodes (%)", 
                Filename = "ProcessDistribution_" + Title1)
    
        for i in nodes_dict:
            Hist = TH1F(Title = i, 
                    xData = nodes_dict[i], 
                    Alpha = 0.5, 
                    xBinCentering = True, 
                    Filename = Title1 + "_" + i)
            Hist.SaveFigure(TargetDir + "SampleStatistics/Raw")
            Hist.Color = None
            H2.Histograms.append(Hist)
        H2.SaveFigure(TargetDir + "SampleStatistics")


    all_, val_, tr_ = {}, {}, {}
    all_n, val_n, tr_n = [], [], []
    for i in DataContainer:
        if DataContainer[i] == None:
            continue
        obj = DataContainer[i]
    
        n_nd = int(obj.num_nodes)
        indx = int(obj.i) 
        
        smpl_ = FileLookup(All, indx)
        if smpl_ not in all_:
            all_[smpl_] = []
            val_[smpl_] = []
            tr_[smpl_] = []
        
        if i in TrainingSample:
            tr_[smpl_].append(n_nd)
            tr_n.append(n_nd)

        if i in ValidationSample:
            val_[smpl_].append(n_nd)
            val_n.append(n_nd)

        all_[smpl_].append(n_nd)
        all_n.append(n_nd)
  

    MakePlot(all_, all_n, "All", "Complete")
    MakePlot(tr_, tr_n, "Training", "Training")
    MakePlot(val_, val_n, "Validation", "Validation")

    text = []
    text.append("==== Complete Sample Composition ====")
    for i in all_:
        f = round(float(len(all_[i])/len(all_n))*100, 3)
        text.append("-> " + i + " | " + str(f) + "%" + " | " + str(len(all_[i])) + " | " + str(len(all_n)))
    
    text.append("")
    text.append("==== Training Sample Composition ====")
    for i in tr_:
        f = round(float(len(tr_[i])/len(tr_n))*100, 3)
        text.append("-> " + i + " | " + str(f) + "%" + " | " + str(len(tr_[i])) + " | " + str(len(tr_n)))
    
    text.append("")
    text.append("==== Validation Sample Composition ====")
    for i in val_:
        f = round(float(len(val_[i])/len(val_n))*100, 3)
        text.append("-> " + i + " | " + str(f) + "%" + " | " + str(len(val_[i])) + " | " + str(len(val_n)))

    F = open(TargetDir + "/SampleStatistics/SampleDecomposition.txt", "w") 
    F.write("\n".join(text))
    F.close()
