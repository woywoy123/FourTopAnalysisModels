from AnalysisTopGNN.IO import UnpickleObject, Directories, WriteDirectory
from AnalysisTopGNN.Generators import EventGenerator, TorchScriptModel, Optimizer
import os, random

from Compilers import *


class MetricsCompiler(EventGenerator, Optimizer):
    def __init__(self):
        super(MetricsCompiler, self).__init__()
        self.FileEventIndex = {}
        self.Device = None
        self.chnk = 10
        self.Threads = 1
        self.DataCacheDir = None

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


    def __EpochLoop(self, Epoch, Statistics, model):
        def UnNest(inpt):
            if isinstance(inpt, list) == False:
                return [inpt]
            out = []
            for i in inpt:
                out += UnNest(i)
            return out
        self.model_dict[model]["EpochTime"] += Statistics["EpochTime"]
        self.model_dict[model]["kFold"] += UnNest(Statistics["kFold"])
        self.model_dict[model]["kFoldTime"] += UnNest(Statistics["FoldTime"])
        
        m_keys = ["Training_Accuracy", "Validation_Accuracy", "Training_Loss", "Validation_Loss"]
        for out in self.model_dict[model]["Outputs"]:
            for k in m_keys:
                self.model_dict[model][out][k.replace("_", "")] += UnNest(Statistics[k][out])
            for k in m_keys[:2]: 
                self.model_dict[model][out][k.split("_")[0] + "Epochs"] += [Epoch for i in range(len(Statistics[k][out]))]
            
        for n, i in zip(Statistics["Nodes"], range(len(Statistics["Nodes"]))):
            if n not in self.model_dict[model]["NodeTime"]:
                self.model_dict[model]["NodeTime"][n] = []
            self.model_dict[model]["NodeTime"][n] += Statistics["FoldTime"][i]
            
            for out in self.model_dict[model]["Outputs"]:
                for k in m_keys:
                    if n not in self.model_dict[model][out]["Node" + k.replace("_", "")]:
                        self.model_dict[model][out]["Node" + k.replace("_", "")][n] = []
                    self.model_dict[model][out]["Node" + k.replace("_", "")][n] += Statistics[k][out][i]

    def ModelTrainingCompiler(self, TrainingStatistics):
        self.model_dict = {}
        GeneralDetails = ["TrainingTime", "Tree", "Start", "End", "Level", "SelfLoop", "Samples", "BatchSize", "Model"]
        for model in TrainingStatistics:
            self.model_dict[model] = {}
            self.model_dict[model] |= { k : TrainingStatistics[model]["Done"][k] for k in GeneralDetails }
            self.model_dict[model]["Outputs"] = list(TrainingStatistics[model]["Done"]["Training_Accuracy"])

            Epochs = [key for key in list(TrainingStatistics[model]) if key != "Done"]
            self.model_dict[model] |= {k : [] for k in ["kFold", "EpochTime", "kFoldTime", "Epochs"]}
            self.model_dict[model]["NodeTime"] = {}
                
            for out in self.model_dict[model]["Outputs"]:
                self.model_dict[model][out] = {}
                self.model_dict[model][out] |= {k : [] for k in ["TrainingLoss", "ValidationLoss"]}
                self.model_dict[model][out] |= {k : [] for k in ["TrainingEpochs", "ValidationEpochs"]}
                self.model_dict[model][out] |= {k : [] for k in ["TrainingAccuracy", "ValidationAccuracy"]}
                self.model_dict[model][out] |= {k : {} for k in ["NodeValidationAccuracy", "NodeTrainingAccuracy"]}
                self.model_dict[model][out] |= {k : {} for k in ["NodeValidationLoss", "NodeTrainingLoss"]}

            for ep in Epochs:
                self.__EpochLoop(ep, TrainingStatistics[model][ep], model)
                self.model_dict[model]["Epochs"] += [ep]
        return self.model_dict

    def __RunPredictions(self, ImportModel, ModelName, Data):
        self._init = False
        self.Model = ImportModel
        self.Sample = Data[0]
        self.InitializeModel()
        for i in Data:
            self.MakePrediction(i)




    def ModelPrediction(self, Torch_Script, Data):
        Data = self.RecallFromCache(Data, self.DataCacheDir) 
        for model in Torch_Script:
            if isinstance(Torch_Script[model], list) == False:
                continue
            Map = Torch_Script[model][0]
            model_dir = Torch_Script[model][1]
            epochs = list(model_dir)
            epochs.sort()
            for epoch in epochs:
                self.Notify("Importing Torch Script Model @ Epoch: " + str(epoch))
                m = TorchScriptModel(model_dir[epoch], maps = Map)
                self.__RunPredictions(m, model, Data) 
                exit()

class ModelEvaluator(EventGenerator, Directories, WriteDirectory):
    
    def __init__(self):
        super(ModelEvaluator, self).__init__()
        self._rootDir = None
        self.Caller = "ModelEvaluator"
        self.Device = "cpu"
        self.DataCacheDir = "HDF5"

        # ==== Compiler Classes
        self._MetricsCompiler = MetricsCompiler()
        self._LogCompiler = LogCompiler()
        self._GraphicsCompiler = GraphicsCompiler() 

        # ==== Internal Sample Information
        self._DataLoaderMap = {}
        self._SampleDetails = {}
        self._TestSample = {}
        self._TrainingSample = {}
        self._SamplesHDF5 = []
        
        # ==== HDF5 Stuff 
        self.RebuildSize = 100
        self.RebuildRandom = True

        # ==== Nodes Compiler 
        self._CompileSampleNodes = False
        self.MakeSampleNodesPlots = True

        # ==== Model Compiler
        self.MakeStaticHistogramPlot = True
        self.MakeTrainingPlots = True

        self._CompileModelOutput = False
        self._CompileModels = False
        self._TrainingStatistics = {}
        self._TorchScripts = {}

    def AddFileTraces(self, Directory):
        if Directory.endswith("FileTraces.pkl"):
            Directory = "/".join(Directory.split("/")[:-1])
        if Directory.endswith("FileTraces") == False:
            Directory = Directory + "/FileTraces"
        self._rootDir = "/".join(Directory.split("/")[:-1])
        self._CompileSampleNodes = True

    def AddModel(self, Directory):
        if Directory.endswith("/"):
            Directory = Directory[:-1]
        
        tmp = self.VerboseLevel
        self.VerboseLevel = 0
        self._TrainingStatistics[Directory.split("/")[-1]] = self.ListFilesInDir(Directory + "/Statistics/", [".pkl"])

        self._TorchScripts[Directory.split("/")[-1]] = self.ListFilesInDir(Directory + "/TorchScript/", [".pt"])
        self._TorchScripts[Directory.split("/")[-1]] = {int(k.split("/")[-1].split("_")[1].split(".")[0]) : k for k in self._TorchScripts[Directory.split("/")[-1]]}
        self.VerboseLevel = tmp

        self.Notify("Added Model: " + Directory.split("/")[-1])
        self._CompileModels = True

    def DefineModelOutputs(self, Name, OutputMap = None):
        if OutputMap == None:
            pt = TorchScriptModel(self._TorchScripts[Name][0])
            pt.ShowNodes()
            self.Fail("OutputMap is not defined. Showing Output Nodes.")
        elif isinstance(OutputMap, list) == False:
            self.Fail("To assign the output, you need to provide a dictionary list. E.g. [{ 'name' : '...', 'node' : '...' }]")
        else:
            pt = TorchScriptModel(self._TorchScripts[Name][0], VerboseLevel = 3, maps = OutputMap)
        
        for i in self._TorchScripts:
            self._TorchScripts[i] = [OutputMap, self._TorchScripts[i]]
        self._CompileModelOutput = True

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
                self._SamplesHDF5.append(name.split(".")[0])
                os.symlink(os.path.abspath(lst[i]), os.path.abspath(self._rootDir + "/HDF5/" + name))
            except FileExistsError:
                pass
            self.Notify("!!!Creating Symlink: " + name)
            if (i+1) % 10000 == 0 or self.VerboseLevel == 3: 
                self.Notify("!!" + str(round(float(i/leng)*100, 3)) + "% Complete")
        self.DataCacheDir = self._rootDir + "/" + self.DataCacheDir + "/"



    def Compile(self, OutDir = "./ModelEvaluator"):

        if self._CompileSampleNodes:
            FileTrace = UnpickleObject(self._rootDir + "/FileTraces/FileTraces.pkl")
            keys = ["Tree", "Start", "End", "Level", "SelfLoop", "Samples"]
            self._SampleDetails |= { key : FileTrace[key] for key in keys}
            self._DataLoaderMap |= FileTrace["SampleMap"]

            TrainSample = UnpickleObject(self._rootDir + "/FileTraces/TrainingSample.pkl")
            self._TrainingSample |= {node : [self._DataLoaderMap[evnt] for evnt in  TrainSample["Training"][node]] for node in  TrainSample["Training"] }
            self._TestSample |= {node : [self._DataLoaderMap[evnt] for evnt in  TrainSample["Validation"][node]] for node in TrainSample["Validation"] }
            dict_stat = self._MetricsCompiler.SampleNodes(TrainSample, FileTrace)

            self._GraphicsCompiler.pwd = OutDir
            self._GraphicsCompiler.MakeSampleNodesPlot = self.MakeSampleNodesPlots
            self._GraphicsCompiler.SampleNodes(dict_stat)
            self._LogCompiler.pwd = OutDir
            self._LogCompiler.SampleNodes(dict_stat)

            self.__BuildSymlinks()

        if self._CompileModels:
            EpochContainer = {}
            for model in self._TrainingStatistics:
                EpochContainer[model] = {k.split("/")[-1].split("_")[1].split(".")[0] : UnpickleObject(k) for k in self._TrainingStatistics[model]}

            model_dict = self._MetricsCompiler.ModelTrainingCompiler(EpochContainer)
            self._GraphicsCompiler.MakeStaticHistograms = self.MakeStaticHistogramPlot
            self._GraphicsCompiler.MakeTrainingPlots = self.MakeTrainingPlots
            
            for model in model_dict:
                self._GraphicsCompiler.pwd = OutDir + "/" + model
                self._GraphicsCompiler.TrainingPlots(model_dict, model)
        
        if self._CompileModelOutput:
            self._MetricsCompiler.Device = self.Device
            self._MetricsCompiler.DataCacheDir = self.DataCacheDir
            self._MetricsCompiler.ModelPrediction(self._TorchScripts, self._SamplesHDF5) 
