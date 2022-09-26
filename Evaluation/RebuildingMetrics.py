from AnalysisTopGNN.IO import UnpickleObject, Directories, WriteDirectory
from AnalysisTopGNN.Generators import EventGenerator
import os, random

from Compilers import *


class MetricsCompiler(EventGenerator):
    def __init__(self):
        super(MetricsCompiler, self).__init__()
        self.FileEventIndex = {}

    def SampleNodes(self, Training, FileTrace):
        self.FileEventIndex |= {"/".join(FileTrace["Samples"][i].split("/")[-2:]) : [FileTrace["Start"][i], FileTrace["End"][i]] for i in range(len(FileTrace["Start"]))}

        RevNode = {ev : [node, True] for node in Training["Training"] for ev in Training["Training"][node]}
        RevNode |= {ev : [node, False] for node in Training["Validation"] for ev in Training["Validation"][node]}
        
        dict_stat = {}
        for evnt in FileTrace["SampleMap"]:
            p_ = self.EventIndexFileLookup(evnt).split("/")[0]
            if p_ not in dict_stat:
                dict_stat[p_] = {"All" : {}, "Training" : {}, "Test" : {} }
            
            n_ = RevNode[evnt][0]
            tr_ = RevNode[evnt][1]
            if n_ not in dict_stat[p_]["All"]:
                dict_stat[p_]["All"][n_] = 0
                dict_stat[p_]["Training"][n_] = 0
                dict_stat[p_]["Test"][n_] = 0

            dict_stat[p_]["All"][n_] += 1
            if tr_:
                dict_stat[p_]["Training"][n_] += 1
            else:
                dict_stat[p_]["Test"][n_] += 1
        return dict_stat


class ModelEvaluator(EventGenerator, Directories, WriteDirectory):
    
    def __init__(self):
        super(ModelEvaluator, self).__init__()
        self._rootDir = None
        self._DataLoaderMap = {}
        self._SampleDetails = {}
        self._TestSample = {}
        self._TrainingSample = {}

        self.RebuildSize = 100
        self.RebuildRandom = True
        self.MakeSampleNodesPlot = True

        self._LogCompiler = LogCompiler()
        self._GraphicsCompiler = GraphicsCompiler() 
        self._MetricsCompiler = MetricsCompiler()

    def AddFileTraces(self, Directory):
        if Directory.endswith("FileTraces.pkl"):
            Directory = "/".join(Directory.split("/")[:-1])
        if Directory.endswith("FileTraces") == False:
            Directory = Directory + "/FileTraces"
        self._rootDir = "/".join(Directory.split("/")[:-1])

    def __BuildSymlinks(self):
        tmp = self.VerboseLevel 
        self.VerboseLevel = 0
        lst = self.ListFilesInDir(self._rootDir + "/", [".hdf5"])
        self.VerboseLevel = tmp

        if self.RebuildSize:
            random.shuffle(lst) 
        
        self.MakeDir(self._rootDir + "/HDF5")
        leng = int(len(lst)*(self.RebuildSize/100))
        for i in range(leng):
            name = lst[i].split("/")[-1]
            try:
                os.symlink(lst[i], self._rootDir + "/HDF5/" + name)
            except FileExistsError:
                pass
            self.Notify("!!!Creating Symlink: " + name)
            if (i+1) % 10000 == 0 or self.VerboseLevel == 3: 
                self.Notify("!!" + str(round(float(i/leng)*100, 3)) + "% Complete")


    def Compile(self, OutDir = "./ModelEvaluator"):
        FileTrace = UnpickleObject(self._rootDir + "/FileTraces/FileTraces.pkl")
        keys = ["Tree", "Start", "End", "Level", "SelfLoop", "Samples"]
        self._SampleDetails |= { key : FileTrace[key] for key in keys}
        self._DataLoaderMap |= FileTrace["SampleMap"]

        TrainSample = UnpickleObject(self._rootDir + "/FileTraces/TrainingSample.pkl")
        self._TrainingSample |= {node : [self._DataLoaderMap[evnt] for evnt in  TrainSample["Training"][node]] for node in  TrainSample["Training"] }
        self._TestSample |= {node : [self._DataLoaderMap[evnt] for evnt in  TrainSample["Validation"][node]] for node in TrainSample["Validation"] }
        dict_stat = self._MetricsCompiler.SampleNodes(TrainSample, FileTrace)

        self._GraphicsCompiler.pwd = OutDir
        self._GraphicsCompiler.MakeSampleNodesPlot = self.MakeSampleNodesPlot
        self._GraphicsCompiler.SampleNodes(dict_stat)
        self._LogCompiler.pwd = OutDir
        self._LogCompiler.SampleNodes(dict_stat)

        self.__BuildSymlinks()
