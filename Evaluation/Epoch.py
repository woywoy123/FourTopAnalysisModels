from AnalysisTopGNN.Tools import Threading, Notification
from AnalysisTopGNN.IO import UnpickleObject
from AnalysisTopGNN.Generators import EventGenerator, GenerateDataLoader, Optimizer, ModelImporter
from glob import glob
import os
import random
import torch

class Tools:
    def UnNestList(self, inpt):
        if isinstance(inpt, list) == False:
            return [inpt]
        out = []
        for i in inpt:
            out += self.UnNestList(i)
        return out

    def UnNestDict(self, inpt):
        if isinstance(inpt, dict) == False:
            return inpt        
        out = []
        for i in inpt:
            out += self.UnNestDict(inpt[i])
        return out 

    def pwd(self):
        return os.getcwd()

    def abs(self, directory):
        return os.path.abspath(directory)

    def mkdir(self, directory):
        try:
            os.makedirs(directory)
        except FileExistsError:
            pass

    def ListFilesInDir(self, directory):
        return [i.split("/")[-1] for i in glob(directory + "/*")]

    def TraverseDictionary(self, inpt, path):
        if isinstance(inpt, dict) == False:
            return inpt
        split = path.split("/")
        if len(split) == 1:
            return inpt[split[0]]
        return self.TraverseDictionary(inpt[split[0]], "/".join(split[1:]))


class Sample(GenerateDataLoader):

    def __init__(self):
        self.src = None
        self.dst = None
        self.prc = None
        self.hash = None
        self.index = None
        self.train = None
        self.Data = None
    
    def Compile(self):
        try:
            os.symlink(self.src + "/" + self.hash + ".hdf5", self.dst + "/" + self.hash + ".hdf5") 
        except FileExistsError:
            pass
        self.Data = self.RecallFromCache(self.hash, self.dst)
        del self.src
        del self.dst

class SampleContainer(Tools, EventGenerator):

    def __init__(self):
        self.DataCache = None
        self.FileTrace = None
        self.TrainingSample = None
        self.HDF5 = None
        self.Threading = 12
        self.chnk = 1000
        self.random = False
        self.Size = 10
        self.Device = "cuda"

    def Collect(self):
        self.FileTrace = UnpickleObject(self.FileTrace)
        self.TrainingSample = UnpickleObject(self.TrainingSample)
        self.SampleMap = self.FileTrace["SampleMap"]

        self.FileEventIndex = ["/".join(smpl.split("/")[-2:]).replace(".root", "") for smpl in self.FileTrace["Samples"]]
        self.FileEventIndex = {self.FileEventIndex[i] : [self.FileTrace["Start"][i], self.FileTrace["End"][i]] for i in range(len(self.FileEventIndex))}
        self.ReverseHash = {str(j) : i for i in self.ListFilesInDir(self.DataCache) for j in UnpickleObject(self.DataCache + "/" + i + "/" + i + ".pkl").values()}

        smpl = list(self.SampleMap) 
        if self.random: 
            random.shuffle(smpl)
        self.SampleMap = {smpl[s] : self.SampleMap[smpl[s]] for s in range(int(len(smpl)*float(self.Size/100)))}

    def MakeSamples(self):
        mode = {i : True for i in self.UnNestDict(self.TrainingSample["Training"])}
        mode |= {i : False for i in self.UnNestDict(self.TrainingSample["Validation"])}
        
        for indx in self.SampleMap:
            smpl = Sample()
            smpl.index = indx
            smpl.hash = self.SampleMap[indx]

            file = self.EventIndexFileLookup(indx)
            sub = file.split("/")
            smpl.prc = sub.pop(0)
            
            smpl.train = mode[indx]

            dch = self.ReverseHash[smpl.hash]
            smpl.src = self.DataCache + "/" + dch + "/" + "/".join(sub)
            smpl.dst = self.HDF5
            smpl.train = False
            
            self.SampleMap[indx] = smpl

        del self.ReverseHash
        del self.DataCache
        del self.FileEventIndex
    
    def Compile(self):
        def Function(inpt):
            for i in inpt:
                i.Compile()
            return inpt
        TH = Threading(list(self.SampleMap.values()), Function, self.Threading, self.chnk)
        TH.Start()
        for i in TH._lists:
            self.SampleMap[i.index] = i
            self.SampleMap[i.index].Data.to(self.Device)


class Epoch(Tools, Optimizer):

    def __init__(self):
        self.Epoch = None
        self.ModelName = None       
        self.Model = None
        self.TorchSave = None
        self.TorchScript = None
        self.ONNX = None
        self.TrainStats = None
        self.ModelInputs = None
        self.ModelOutputs = None
        self.ModelTruth = None
        self.Mode = "Test"
        self.Training = False
        self.Device = None

    def CollectMetric(self, name, key, feature, inpt):
        if hasattr(self, name) == False:
            setattr(self, name, {})
        if feature not in getattr(self, name):
            d = getattr(self, name)
            d[feature] = []
        for i in self.TraverseDictionary(inpt, key):
            d[feature] += self.UnNestList(i)

    def CompileTraining(self):
        self.TrainStats = UnpickleObject(self.TrainStats)
        self.EpochTime = self.TrainStats["EpochTime"][0]

        Metrics = ["Training_Loss", "Training_Accuracy", "Validation_Loss", "Validation_Accuracy"]
        for metric in Metrics:
            for feat in self.TrainStats[metric]:
                self.CollectMetric(metric.replace("_", ""), metric + "/" + feat, feat, self.TrainStats)
      
        self.FoldTime = []
        self.KFolds = []
        for k in range(len(self.TrainStats["kFold"])):
            nodes = self.TrainStats["Nodes"][k]
            self.CollectMetric("NodeTime", "FoldTime", nodes, self.TrainStats)
            
            self.FoldTime += self.TrainStats["FoldTime"][k]
            self.KFolds += self.TrainStats["kFold"][k]
            
            for metric in Metrics:
                for feat in self.TrainStats[metric]:
                    inpt = self.TrainStats[metric][feat][nodes]
                    self.CollectMetric("Node"+metric.replace("_", ""), self.TrainStats[metric], nodes, inpt)
        del self.TrainStats
    
    def PredictInput(self, Data):
        self.Debug = self.Mode
        self.Stats = {}
        self.MakeContainer(self.Mode)
        self.Model.load_state_dict(torch.load(self.TorchSave)["state_dict"])
        self.Device = list(Data.values())[0].Data.device
        for i in Data:
            self.Train(Data[i].Data)



class ModelContainer(Tools, ModelImporter, Notification):

    def __init__(self, Name = None):
        self.Epochs = {}
        self.TorchScriptMap = None
        self.ModelSaves = {
            "TorchSave" : {}, 
            "TorchScript" : {}, 
            "Base" : None, 
            "Training" : {}
        }
        
        self.EdgeFeatures = {}
        self.NodeFeatures = {}
        self.GrapFeatures = {}
        self.T_Features = {}
        
        self.ModelOutputs = {}
        self.ModelInputs = {} 

        self.Name = Name
        self.Data = {}
        self.Dir = None
        self.VerboseLevel = 0
        self.Caller = "ModelContainer"

    def Collect(self):
        Files = self.ListFilesInDir(self.Dir + "/TorchSave")
        
        self.ModelSaves["Base"] = Files.pop(Files.index([i for i in Files if "_Model.pt" in i][0]))
        self.Epochs |= { ep.split("_")[1] : None for ep in Files}
        self.ModelSaves["TorchSave"] |= { ep.split("_")[1] : self.Dir + "/TorchSave/" + ep for ep in Files} 

        Files = self.ListFilesInDir(self.Dir + "/TorchScript")
        self.ModelSaves["TorchScript"] |= { ep.split("_")[1] : self.Dir + "/TorchScript/" + ep for ep in Files} 
        
        Files = self.ListFilesInDir(self.Dir + "/Statistics")
        self.ModelSaves["Training"] |= { ep.split("_")[1].replace(".pkl", "") : self.Dir + "/Statistics/" + ep for ep in Files}
    
    def MakeEpochs(self):
        for ep in self.Epochs:
            self.Epochs[ep] = Epoch()
            self.Epochs[ep].Epoch = ep
            self.Epochs[ep].ModelName = self.Name
            self.Epochs[ep].TrainStats = self.ModelSaves["Training"][ep]
            self.Epochs[ep].TorchSave = self.ModelSaves["TorchSave"][ep]
            self.Epochs[ep].TorchScript = self.ModelSaves["TorchScript"][ep]
            self.Epochs[ep].Device = self.Device
    
    def AnalyzeDataCompatibility(self):
        if len(self.Data) == 0:
            return False
        self._init = False
        self.Model = torch.load(self.Dir + "/TorchSave/" + self.ModelSaves["Base"])
        self.Sample = list(self.Data.values())[0].Data
        self.InitializeModel()
        self.GetTruthFlags(FEAT = "E")
        self.GetTruthFlags(FEAT = "N")
        self.GetTruthFlags(FEAT = "G")
        
        for i in self.Epochs:
            self.Epochs[i].ModelInputs = self.ModelInputs
            self.Epochs[i].ModelOutputs = self.ModelOutputs
            self.Epochs[i].T_Features = self.T_Features
            self.Epochs[i].Model = self.Model
        
    def CompileTrainingStatistics(self):
        for ep in self.Epochs:
            self.Epochs[ep].CompileTraining()

    def CompileTestData(self):
        for ep in self.Epochs:
            self.Epochs[ep].PredictInput(self.Data)





