from AnalysisTopGNN.IO import UnpickleObject, PickleObject
from AnalysisTopGNN.Generators import Optimizer
from Tooling import Tools, Metrics
import torch

class Epoch(Tools, Optimizer, Metrics):

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
        self.Training = False
        self.Device = None
        
        self.Debug = False
        self.Stats = {}
        self.ROC = {}
        
        self.NodeFeatureMass = {}
        self.EdgeFeatureMass = {}

        self.TruthNodeFeatureMass = {}
        self.TruthEdgeFeatureMass = {}

        self.NodeParticleEfficiency = {}
        self.EdgeParticleEfficiency = {}

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
    
    def PredictOutput(self, Data, idx):
        truth, pred = self.Train(Data[idx].Data)
        for feat in list(pred):
            if feat not in self.ROC:
                self.ROC[feat] = { "fpr" : [], "tpr" : [], "auc" : [], 
                                   "truth" : [], "pred" : [], "pred_score" : [], 
                                   "idx" : [], "proc" : []
                                 }

            self.ROC[feat]["truth"].append(truth[feat][0])
            self.ROC[feat]["pred"].append(pred[feat][0])
            self.ROC[feat]["pred_score"].append(pred[feat][1]) 
            self.ROC[feat]["idx"].append(idx)
            self.ROC[feat]["proc"].append(Data[idx].prc)
        return pred

    def Flush(self):
        self.Model.load_state_dict(torch.load(self.TorchSave)["state_dict"])
        self.Stats = {}
        self.ROC = {}
        self.NodeFeatureMass = {}
        self.EdgeFeatureMass = {}
        self.TruthNodeFeatureMass = {}
        self.TruthEdgeFeatureMass = {}
        self.NodeParticleEfficiency = {}
        self.EdgeParticleEfficiency = {}
        if self.Debug:
            self.MakeContainer(self.Debug)

    def ParticleYield(self, Edge):
        dic_p = self.EdgeFeatureMass if Edge else self.NodeFeatureMass
        dic_t = self.TruthEdgeFeatureMass if Edge else self.TruthNodeFeatureMass
        dic_o = self.EdgeParticleEfficiency if Edge else self.NodeParticleEfficiency

        for pack in [[feat, event, prc] for feat in dic_p for event, prc in zip(dic_p[feat], self.ROC[feat]["proc"])]:
            f, idx, prc = pack[0], pack[1], pack[2]
            if f not in dic_o:
                dic_o[f] = {}
            dic_o[f][idx] = self.ParticleEfficiency(dic_p[f][idx], dic_t[f][idx], prc)

    
    def DumpEpoch(self, ModeType, OutputDir):
        def ParticleDumping(Pred_Mass, Truth_Mass, Effic, Key):
            Output = {}
            for i in Pred_Mass:
                Output[Key + "ParticleMass/"+ i + "/MassDistribution_TH1FStack|Truth|Prediction.GeV.Entries"] = {"|Truth" : self.UnNestDict(Truth_Mass[i]), "|Prediction" : self.UnNestDict(Pred_Mass[i])}
                
                prc = self.CollectKeyNestDict(Effic[i], "Prc")
                per = self.CollectKeyNestDict(Effic[i], "%")
                Output[Key + "ParticleMass/"+ i + "/ProcessReconstructionEfficiency_Point.Epoch.%"] = {p : [ per[k] for k in range(len(prc)) if prc[k] == p ] for p in list(set(prc))}
    
                ntru = self.CollectKeyNestDict(Effic[i], "ntru")
                Output[Key + "ParticleMass/"+ i + "/SampleProcessComposition_TH1FStack|Truth|Predicted.n-Particles.Entries"] = {p + "|Truth" : [ ntru[k] for k in range(len(prc)) if prc[k] == p ] for p in list(set(prc))}
    
                pred = self.CollectKeyNestDict(Effic[i], "nrec")
                Output[Key + "ParticleMass/"+ i + "/SampleProcessComposition_TH1FStack|Truth|Predicted.n-Particles.Entries"] |= {p + "|Predicted" : [ pred[k] for k in range(len(prc)) if prc[k] == p ] for p in list(set(prc))}
    
                Output[Key + "ParticleMass/" + i + "/AllCollectedParticles_Point.Epoch.%"] = (float(sum(pred)/sum(ntru)))*100
            return Output
        def ROCDumping(ROC_Dict):
            Output = {}
            Title = "ROC/CombinedFeatures_ROC|AUC=.False Positive Rate.True Positive Rate"
            Output[Title] = {}
            Output["AUC/AllCollected_Point.Epoch.AUC"] = {}
            for feat in ROC_Dict:
                Output[Title] |= { feat + "|AUC=" + str(round(ROC_Dict[feat]["auc"][0], 3)) : {"False Positive Rate" : ROC_Dict[feat]["fpr"], "True Positive Rate" : ROC_Dict[feat]["tpr"]} }
                Output["AUC/AllCollected_Point.Epoch.AUC"] |= {feat : ROC_Dict[feat]["auc"]}
            return Output

        def DumpSampleLoss(Feat, LossDict):
            return {"Loss/" + Feat + "/LossPlot_Point.Epoch.Loss" : LossDict[Feat]}

        def DumpSampleAccuracy(Feat, AccDict):
            return {"Accuracy/" + Feat + "/AccuracyPlot_Point.Epoch.Accuracy (%)" : AccDict[Feat]}
        
        def DumpTime():
            Output = {
                        "Time/EpochTime_Point.Epoch.Time (s)" : self.EpochTime, 
                        "Time/NodeTime_TH1FStack.Nodes.Time (s)" : self.NodeTime
                     }
            return Output
        
        self.mkdir(OutputDir + "/" + self.ModelName + "/" + ModeType + "/Epochs/")
        Out = {}
        Out |= ParticleDumping(self.EdgeFeatureMass, self.TruthEdgeFeatureMass, self.EdgeParticleEfficiency, "Edge")
        Out |= ParticleDumping(self.NodeFeatureMass, self.TruthNodeFeatureMass, self.NodeParticleEfficiency, "Node")
        Out |= ROCDumping(self.ROC)
        
        for metric in self.Stats:
            for feat in self.Stats[metric]:
                Out |= DumpSampleAccuracy(feat, self.Stats[metric]) if "_Accuracy" in metric else DumpSampleLoss(feat, self.Stats[metric])
       
        for key in self.__dict__:
            val = self.__dict__[key]
            if key.startswith("Node"):
                continue
            if key.endswith("Loss"):
                Out[key.split("Loss")[0]] = {}
                for feat in val:
                    Out[key.split("Loss")[0]] |= DumpSampleLoss(feat, val)
            elif key.endswith("Accuracy"):
                Out[key.split("Accuracy")[0]] = {}
                for feat in val:
                    Out[key.split("Accuracy")[0]] |= DumpSampleLoss(feat, val)
        
        if len(self.Stats) == 0:
            Out |= DumpTime()
        PickleObject(Out, str(self.Epoch), OutputDir + "/" + self.ModelName + "/" + ModeType + "/Epochs/") 
        self.Flush()
