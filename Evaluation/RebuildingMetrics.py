from AnalysisTopGNN.IO import UnpickleObject, Directories, WriteDirectory, PickleObject
from AnalysisTopGNN.Generators import EventGenerator, TorchScriptModel, Optimizer
import os, random
import torch

from Compilers import *


class MetricsCompiler(EventGenerator, Optimizer):
    def __init__(self):
        super(MetricsCompiler, self).__init__()
        self.FileEventIndex = {}
        self.Device = None
        self.chnk = 10
        self.Threads = 1
        self.DataCacheDir = None
        self.CompareToTruth = False
        self.Predictions = {}
        self.Truths = {}
        self.Statistics = {}
        self.model_dict = {}
        self.model_test = {}
        self.ROC_dict = {}

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


    def UnNest(self, inpt):
        if isinstance(inpt, list) == False:
            return [inpt]
        out = []
        for i in inpt:
            out += self.UnNest(i)
        return out

    def __EpochLoopGeneral(self, Epoch, Statistics, model, m_keys, model_dict, metrics = ["Accuracy", "Loss"]):

        for out in model_dict[model]["Outputs"]:
            for k in [t + "_" + m for t in m_keys for m in metrics]:
                model_dict[model][out][k.replace("_", "")] += self.UnNest(Statistics[k][out])
            for k in m_keys: 
                model_dict[model][out][k + "Epochs"] += [Epoch for i in range(len(self.UnNest(Statistics[k + "_" + metrics[0]][out])))]
    
    def __EpochLoopNodes(self, Epoch, Statistics, model, m_keys, model_dict):
        for n, i in zip(Statistics["Nodes"], range(len(Statistics["Nodes"]))):
            if n not in model_dict[model]["NodeTime"]:
                model_dict[model]["NodeTime"][n] = []
            model_dict[model]["NodeTime"][n] += Statistics["FoldTime"][i]
            
            for out in model_dict[model]["Outputs"]:
                for k in m_keys:
                    if n not in model_dict[model][out]["Node" + k.replace("_", "")]:
                        model_dict[model][out]["Node" + k.replace("_", "")][n] = []
                    model_dict[model][out]["Node" + k.replace("_", "")][n] += Statistics[k][out][i]

    def ModelTrainingCompiler(self, TrainingStatistics):
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
                Statistics = TrainingStatistics[model][ep]
                self.model_dict[model]["EpochTime"] += Statistics["EpochTime"]
                self.model_dict[model]["kFold"] += self.UnNest(Statistics["kFold"])
                self.model_dict[model]["kFoldTime"] += self.UnNest(Statistics["FoldTime"])
                
                self.__EpochLoopGeneral(ep, Statistics, model, ["Training", "Validation"], self.model_dict)
                self.__EpochLoopNodes(ep, Statistics, model, ["Training_Accuracy", "Validation_Accuracy", "Training_Loss", "Validation_Loss"], self.model_dict)
                self.model_dict[model]["Epochs"] += [ep]
        return self.model_dict

    def ModelPrediction(self, TorchSave, DataInput, TorchScript):
        Data = self.RecallFromCache(DataInput, self.DataCacheDir) 
        self.Training = True

        self.DefaultOptimizer = "ADAM"
        self.LearningRate = 0.001
        self.WeightDecay = 0.001
        
        self.Debug = "Test"
        for model in TorchSave:
            epochs = list(TorchSave[model])
            epochs.sort()
            _zero = epochs.pop(0)

            self._init = False
            self.Sample = Data[0]
            self.Model = torch.load(TorchSave[model][_zero])
            self.InitializeModel()
            self.Predictions[model] = {}
            self.Predictions[model] |= {ep : {} for ep in epochs}
            
            self.DefineOptimizer()
            self.T_Features = {}
            self.GetTruthFlags([], "E")
            self.GetTruthFlags([], "N")
            self.GetTruthFlags([], "G")
            self.Statistics[model] = {}
            self.Statistics[model]["Outputs"] = [k[2:] for k in self.ModelOutputs if k.startswith("O_")]
            for epoch in epochs:
                if model in TorchScript:
                    self.Model = TorchScriptModel(TorchScript[model][1][epoch], maps = TorchScript[model][0])
                    self.Model.to(self.Device)
 
                self.Stats = {}
                self.MakeContainer("Test")

                self.Notify("Importing Model: '" + model + "' @ Epoch: " + str(epoch))
                output = {}
                for d in Data:
                    try:
                        truth, pred = self.Train(d)
                    except:
                        self.Warning("Imported TorchScript Failed. Revert to Pickled Model.")
                        self.Model = torch.load(TorchSave[model][epoch])
                        self.Model.to(self.Device)
                        truth, pred = self.Train(d) 
                    it = d.i.item()
                
                    output[it] = {v : pred[v][0] for v in pred}
                    if self.CompareToTruth and it not in self.Truths:
                        self.Truths[it] = {v : truth[v][0] for v in truth}
                    self.Predictions[model][epoch] |= output 
                self.Statistics[model][epoch] = self.Stats
    
    def ModelTestCompiler(self, Statistics):
        for model in Statistics:
            self.model_test[model] = {}
            self.model_test[model]["Outputs"] = Statistics[model]["Outputs"]
            for out in self.model_test[model]["Outputs"]:
                self.model_test[model][out] = {}
                self.model_test[model][out]["TestLoss"] = []
                self.model_test[model][out]["TestAccuracy"] = []
                self.model_test[model][out]["TestEpochs"] = []

            for Epoch in [i for i in list(Statistics[model]) if isinstance(i, int)]:
                Stats = Statistics[model][Epoch]
                self.__EpochLoopGeneral(Epoch, Stats, model, ["Test"], self.model_test)
        return self.model_test

    def ROCCurveCompiler(self, pred, truth, model, features):
        for epoch in pred[model]:
            for f in features:
                if f not in self.ROC_dict:
                    self.ROC_dict[f] = {}
                if epoch not in self.ROC_dict[f]:
                    self.ROC_dict[f][epoch] = {}
                if model not in self.ROC_dict[f][epoch]:
                    self.ROC_dict[f][epoch][model] = {"TP" : [], "TN" : [], "FP" : [], "FN" : []}
                    
                for ev in pred[model][epoch]:
                    t = truth[ev][f]
                    p = pred[model][epoch][ev][f]
                    
                    p_1, t_1 = p == 1, t == 1
                    p_0, t_0 = p == 0, t == 0
                    
                    pt_11 = torch.sum(p_1 == t_1)
                    pt_01 = torch.sum(p_0 == t_1)
                    pt_00 = torch.sum(p_0 == t_0)
                    
                    tpr = pt_11 / (pt_11 + pt_01)
                    fnr = 1-tpr
                # Can be easily solved via CUDA 
                # Same with the invariant mass 



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
        self.CompareToTruth = False
        self._CompileModelOutput = False
        self._CompileModels = False
        
        self._ROCCurveFeatures = {}
        self._TrainingStatistics = {}
        self._TorchScripts = {}
        self._TorchSave = {}

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
        Name = Directory.split("/")[-1]

        tmp = self.VerboseLevel
        self.VerboseLevel = 0
        x = self.ListFilesInDir(Directory + "/Statistics/", [".pkl"])
        if len(x) == 0:
            self.Warning("Model: " + Name + " not found. Skipping")
            return 
        self._TrainingStatistics[Name] = x
        self._TorchSave[Name] = self.ListFilesInDir(Directory + "/TorchSave/", [".pt"])
        self._TorchSave[Name] = {int(k.split("/")[-1].split("_")[1].split(".")[0]) : k for k in self._TorchSave[Name]}
        
        self.VerboseLevel = tmp
        self.Notify("Added Model: " + Directory.split("/")[-1])
        self._CompileModels = True

    def AddTorchScriptModel(self, Name, OutputMap = None, Directory = None):
        if Directory is None:
            Directory = self._rootDir + "/Models/" + Name + "/TorchScript/"
        self._TorchScripts[Name] = self.ListFilesInDir(Directory, [".pt"])
        self._TorchScripts[Name] = {int(k.split("/")[-1].split("_")[1].split(".")[0]) : k for k in self._TorchScripts[Name]}
        if len(self._TorchScripts[Name]) == 0:
            self.Warning(Name + " not found. Skipping")
            return 
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

    def ROCCurveFeature(self, feature):
        self._ROCCurveFeatures[feature] = None

    def __BuildSymlinks(self):
        tmp = self.VerboseLevel 
        self.VerboseLevel = 0
        lst = [i + ".hdf5" for i in self._DataLoaderMap.values()]
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
                continue
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
        
        if self._CompileModelOutput or len(self._TorchSave) > 0:
            if len(self._SamplesHDF5) == 0:
                self.Fail("No Samples Found.")
            #self._MetricsCompiler.Device = self.Device
            #self._MetricsCompiler.DataCacheDir = self.DataCacheDir
            #self._MetricsCompiler.CompareToTruth = self.CompareToTruth
            #self._MetricsCompiler.ModelPrediction(self._TorchSave, self._SamplesHDF5, self._TorchScripts) 
            #
            #pred = self._MetricsCompiler.Predictions
            #truth = self._MetricsCompiler.Truths
            #stat = self._MetricsCompiler.Statistics
            #
            #PickleObject([stat, pred, truth], "TMP")
            tmp = UnpickleObject("TMP")
            stat = tmp[0]
            
            #stat_dict = self._MetricsCompiler.ModelTestCompiler(stat)
            #for model in stat_dict:
            #    self._GraphicsCompiler.pwd = OutDir + "/" + model 
            #    self._GraphicsCompiler.TestPlots(stat_dict, model)
            
            truth, pred = tmp[2], tmp[1]
            
            for model in pred:
                features = [f for f in self._ROCCurveFeatures if f in stat[model]["Outputs"]]
                if len(features) == 0:
                    continue
                self._MetricsCompiler.ROCCurveCompiler(pred, truth, model, features)
