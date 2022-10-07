import os
from glob import glob
from sklearn.metrics import roc_curve, auc
import numpy as np
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
    
    def CollectKeyNestDict(self, inpt, search):
        if isinstance(inpt, dict) == False:
            return []
        out = []
        for i in inpt:
            if i == search:
                return inpt[i] if isinstance(inpt[i], list) else [inpt[i]]
            out += self.CollectKeyNestDict(inpt[i], search)
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

    def SortEpoch(self, inpt):
        epochs = [int(x) for x in inpt]
        epochs.sort()
        tmp = {}
        tmp |= inpt
        inpt.clear()
        for ep in epochs:
            inpt[ep] = tmp[ep]

class Metrics:

    def MakeROC(self, feature):
        truth = self.ROC[feature]["truth"]
        truth = torch.cat(truth, dim = 0).view(-1)
        truth = truth.detach().cpu().numpy()
        
        p_score = self.ROC[feature]["pred_score"]
        p_score = torch.cat([p.softmax(dim = 1).max(1)[0] for p in p_score], dim = 0)
        p_score = p_score.detach().cpu().numpy()
        
        fpr, tpr, _ = roc_curve(truth, p_score)
        auc_ = auc(fpr, tpr)

        self.ROC[feature]["fpr"] += fpr.tolist()
        self.ROC[feature]["tpr"] += tpr.tolist()
        self.ROC[feature]["auc"].append(float(auc_))

        return self.ROC
   

    def ClosestParticle(self, tru, pred):

        res = []
        if len(tru) == 0:
            return res
        p = pred.pop(0)
        max_tru, min_tru = max(tru), min(tru)
        col = True if p <= max_tru and p >= min_tru else False

        if col == False:
            if len(pred) == 0:
                return res
            return self.ClosestParticle(tru, pred)

        diff = [ abs(p - t) for t in tru ]
        tru.pop(diff.index(min(diff)))
        res += self.ClosestParticle(tru, pred)
        res.append(p)
        return res 
    
    def ParticleEfficiency(self, pred, truth, proc):
        t_, p_ = [], []
        t_ += truth
        p_ += pred 

        p = self.ClosestParticle(t_, p_)
        p_l, t_l = len(p), len(truth)

        perf = float(p_l/t_l)*100

        out = {"Prc" : proc, "%" : perf, "nrec" : p_l, "ntru" : t_l}
        
        return out



from AnalysisTopGNN.Plotting import CombineTLine, TLine, TH1F, CombineTH1F
from AnalysisTopGNN.Tools import Notification
class Template(Tools, Notification):
    
    def __init__(self):
        pass

    def AddPoint(self, Src, Trgt, Epoch):
        for feat in Src:
            if feat not in Trgt:
                Trgt[feat] = {}
            Trgt[feat][Epoch] = Src[feat]

    def TemplateConfig(self, OutputDirectory):
        Plots = {
                    "Filename" : "", 
                    "Style" : "ATLAS", 
                    "OutputDirectory" : OutputDirectory
                }
        return Plots 

    def TemplateTLine(self, OutputDirectory):
        Plots = self.TemplateConfig(OutputDirectory)
        Plots["xData"] = []
        Plots["yData"] = []
        return Plots 

    def TemplateROC(self, OutputDirectory, fpr, tpr):
        Plots = self.TemplateTLine(OutputDirectory)
        Plots["xTitle"] = "False Positive Rate"
        Plots["yTitle"] = "True Positive Rate"
        Plots["xData"] = fpr
        Plots["yData"] = tpr
        Plots["xMin"] = 0
        Plots["yMin"] = 0 
        Plots["xMax"] = 1
        Plots["yMax"] = 1
        return Plots

    def MakeLossPlot(self, InptDic, Feature, Title, LineStyle = None):
        Config = self.TemplateTLine(self.OutputDirectory)
        Config["xTitle"] = "Epoch"
        Config["yTitle"] = "Loss (a.u.)"
        Config["LineStyle"] = LineStyle
        Config["Title"] = Title
        self.SortEpoch(InptDic[Feature])
        Config["xData"] = InptDic[Feature]
        Config["DoStatistics"] = True
        InptDic[Feature] = TLine(**Config)

    def MakeAccuracyPlot(self, InptDic, Feature, Title, LineStyle = None):
        Config = self.TemplateTLine(self.OutputDirectory)
        Config["xTitle"] = "Epoch"
        Config["yTitle"] = "Accuracy (%)"
        Config["Title"] = Title
        Config["LineStyle"] = LineStyle
        self.SortEpoch(InptDic[Feature])
        Config["xData"] = InptDic[Feature]
        Config["DoStatistics"] = True
        InptDic[Feature] = TLine(**Config)

    def MergePlots(self, inptList):
        Config = self.TemplateTLine(self.OutputDirectory)
        Config["Lines"] = inptList
        com = CombineTLine(**Config)
        com.Lines = inptList
        return com

    def AddEpochEdge(self, InptDic, epoch):
        for i in set([k for k in InptDic if k.split("/")[0] == "EdgeParticleMass"]):
            feat = i.split("/")[1]
            if feat not in self.EdgeMass:
                self.EdgeMass[feat] = {}
                self.EdgeMassPrcEff[feat] = {}
                self.EdgeMassPrcCompo[feat] = {}
                self.EdgeMassAll[feat] = {}
            self.EdgeMass[feat][epoch] = InptDic["EdgeParticleMass/" + feat + "/MassDistribution"]
            self.EdgeMassPrcEff[feat][epoch] = InptDic["EdgeParticleMass/" + feat + "/ProcessEfficiency"]
            self.EdgeMassPrcCompo[feat][epoch] = InptDic["EdgeParticleMass/" + feat + "/SampleComposition"]
            self.EdgeMassAll[feat][epoch] = InptDic["EdgeParticleMass/" + feat + "/AllCollectedParticles"]

    def AddEpochNode(self, InptDic, epoch):
        for i in set([k for k in InptDic if k.split("/")[0] == "NodeParticleMass"]):
            feat = i.split("/")[1]
            if feat not in self.NodeMass:
                self.NodeMass[feat] = {}
                self.NodeMassPrcEff[feat] = {}
                self.NodeMassPrcCompo[feat] = {}
                self.NodeMassAll[feat] = {}
            self.NodeMass[feat][epoch] = InptDic["NodeParticleMass/" + feat + "/MassDistribution"]
            self.NodeMassPrcEff[feat][epoch] = InptDic["NodeParticleMass/" + feat + "/ProcessEfficiency"]
            self.NodeMassPrcCompo[feat][epoch] = InptDic["NodeParticleMass/" + feat + "/SampleComposition"]
            self.NodeMassAll[feat][epoch] = InptDic["NodeParticleMass/" + feat + "/AllCollectedParticles"]
    
    def MakeTH1FTemplate(self, OutputDirectory, xData, xTitle, yTitle, Title):
        config = self.TemplateConfig(OutputDirectory)
        config["xTitle"] = xTitle
        config["yTitle"] = yTitle
        config["xData"] = xData
        config["Title"] = Title
        config["xBins"] = 500
        config["xMin"] = 0
        config["xMax"] = 2000
        return TH1F(**config) 
 
    def MergeTH1F(self, inptList, Title):
        Config = self.TemplateConfig(self.OutputDirectory)
        Config["Histograms"] = inptList
        Config["xBins"] = 500
        Config["xMin"] = 0
        Config["xMax"] = 2000
        Config["Logarithmic"] = True
        return CombineTH1F(**Config)

    def MakeMassPlot(self, inpt, mode, output):
        truthvector = {}
        outvector = {}
        for feat in inpt:
            self.SortEpoch(inpt[feat]) 
            for ep in inpt[feat]:   
                if feat not in truthvector:
                    truthvector[feat] = self.MakeTH1FTemplate(output, inpt[feat][ep]["Truth"], "Mass (GeV)", "Entries", "Truth-"+feat)
                if ep not in outvector:
                    outvector[ep] = []
                outvector[ep] += [self.MakeTH1FTemplate(output, inpt[feat][ep]["Prediction"], "Mass (GeV)", "Entries", feat)]
                outvector[ep] += [truthvector[feat]]
        
        for i in outvector:
            m = self.MergeTH1F(outvector[i], "Reconstructed Particle Mass from Graph Neural Network " + mode +" Features at Epoch: " + str(i))
            m.OutputDirectory = output
            m.Filename = "RecoParticleMass-Epoch_" + str(i)
            m.SaveFigure()
    
    def MakeReconstructionProcessEfficiency(self, inpt):
        for feat in inpt:
            self.SortEpoch(inpt[feat])
            epochs = list(inpt[feat])
            prc = list(inpt[feat][epochs[0]])
            prc = { p : [] for p in prc }
            for p in prc:
                lst = {ep : inpt[feat][ep][p] for ep in epochs }
                plt = self.TemplateTLine(self.OutputDirectory)
                plt["xData"] = lst
                plt["DoStatistics"] = True
                plt["Title"] = p
                plt["xTitle"] = "Epoch"
                plt["yTitle"] = "Reconstruction Efficiency (%)"
                prc[p] = TLine(**plt)

            plt = self.MergePlots(list(prc.values()))
            plt.Title = "Reconstruction Efficiency of Processes using Feature " + feat
            plt.Filename = "ProcessReconstruction_" + feat
            plt.yMax = 101
            plt.yMin = -1
            plt.SaveFigure()
 
    def MakeReconstructionEfficiency(self, inpt):
        for feat in inpt:
            self.SortEpoch(inpt[feat])
            plt = self.TemplateTLine(self.OutputDirectory)
            plt["xData"] = list(inpt[feat])
            plt["yData"] = list(inpt[feat].values())
            plt["Title"] = feat
            plt["xTitle"] = "Epoch"
            plt["yTitle"] = "Reconstruction Efficiency (%)"
            inpt[feat] = TLine(**plt)
        if len(inpt) == 0:
            return 
        plt = self.MergePlots(list(inpt.values()))
        plt.Title = "Reconstruction Efficiency of Edge Features"
        plt.Filename = "ProcessReconstructionAll"
        plt.yMax = 101
        plt.yMin = -1
        plt.SaveFigure()
        


