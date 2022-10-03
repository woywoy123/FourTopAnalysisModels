from AnalysisTopGNN.Generators import GenerateDataLoader, EventGenerator
from AnalysisTopGNN.Tools import Threading
from AnalysisTopGNN.IO import UnpickleObject
from Tooling import Tools
import random
import os

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


