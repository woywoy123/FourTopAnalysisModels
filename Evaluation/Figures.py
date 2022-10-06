from AnalysisTopGNN.Plotting.TemplateHistograms import TH1FStack
from AnalysisTopGNN.Plotting.TemplateLines import TLineStack, TLine, CombineTLine
from Tooling import Tools



class Template(Tools):
    
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
                    "yTitle" : "", 
                    "xTitle" : "", 
                    "OutputDirectory" : OutputDirectory
                }
        return Plots 

    def TemplateTLine(self, OutputDirectory):
        Plots = self.TemplateConfig(OutputDirectory)
        Plots["xData"] = []
        Plots["yData"] = []
        return Plots 
    
    def SortEpoch(self, inpt):
        epochs = list(inpt)
        epochs.sort()
        tmp = {}
        tmp |= inpt
        inpt.clear()
        for ep in epochs:
            inpt[ep] = tmp[ep]

    def MakeLossPlot(self, InptDic, Feature, Title, Marker = None):
        Config = self.TemplateTLine(self.OutputDirectory)
        Config["xTitle"] = "Epoch"
        Config["yTitle"] = "Loss (a.u.)"
        Config["Marker"] = Marker
        Config["Title"] = Title
        self.SortEpoch(InptDic[Feature])
        Config["xData"] = InptDic[Feature]
        Config["DoStatistics"] = True
        InptDic[Feature] = TLine(**Config)

    def MakeAccuracyPlot(self, InptDic, Feature, Title, Marker = None):
        Config = self.TemplateTLine(self.OutputDirectory)
        Config["xTitle"] = "Epoch"
        Config["yTitle"] = "Accuracy (%)"
        Config["Title"] = Title
        Config["Marker"] = Marker
        self.SortEpoch(InptDic[Feature])
        Config["xData"] = InptDic[Feature]
        Config["DoStatistics"] = True
        InptDic[Feature] = TLine(**Config)

    def MergePlots(self, inptList):
        for i in inptList:
            i.Compile()
        Config = self.TemplateTLine(self.OutputDirectory)
        com = CombineTLine(**Config)
        com.Lines = inptList
        return com
        



class Training(Template):
    
    def __init__(self):

        self.ValidLoss = {}
        self.ValidAcc = {}

        self.TrainLoss = {}
        self.TrainAcc = {}
        
        self.Epochs = []
        self.EpochTime = []
        self.kFolds = []
        self.FoldTime = []

        self.NodeTime = {}

        self.NodeValidLoss = {}
        self.NodeValidAcc = {}

        self.NodeTrainLoss = {}
        self.NodeTrainAcc = {}

    def AddEpoch(self, epoch, dic):
        self.AddPoint(dic["TrainingLoss"], self.TrainLoss, epoch)
        self.AddPoint(dic["TrainingAccuracy"], self.TrainAcc, epoch)
        self.AddPoint(dic["ValidationLoss"], self.ValidLoss, epoch)
        self.AddPoint(dic["ValidationAccuracy"], self.ValidAcc, epoch)
        self.Epochs.append(epoch)

        self.EpochTime.append(dic["EpochTime"])
        self.FoldTime += dic["FoldTime"]
        self.kFolds += dic["KFolds"]
        self.AddPoint(dic["NodeTime"], self.NodeTime, epoch) 

        self.AddPoint(dic["NodeTrainingLoss"], self.NodeTrainLoss, epoch)
        self.AddPoint(dic["NodeTrainingAccuracy"], self.NodeTrainAcc, epoch)
        self.AddPoint(dic["NodeValidationLoss"], self.NodeValidLoss, epoch)
        self.AddPoint(dic["NodeValidationAccuracy"], self.NodeValidAcc, epoch)
   
    def Compile(self, Output):
        for feat in self.TrainLoss:
            self.OutputDirectory = Output + "/training/plots/loss"
            self.MakeLossPlot(self.TrainLoss, feat, "Training", ".")
            self.MakeLossPlot(self.ValidLoss, feat, "Validation", ",")

            comb = self.MergePlots([self.TrainLoss[feat], self.ValidLoss[feat]])
            comb.Title = "Loss for Feature: " + feat
            comb.Filename = "Loss_" + feat
            comb.SaveFigure()

            self.OutputDirectory = Output + "/training/plots/accuracy"
            self.MakeAccuracyPlot(self.TrainAcc, feat, "Training", ".")
            self.MakeAccuracyPlot(self.ValidAcc, feat, "Validation", ",")

            comb = self.MergePlots([self.TrainAcc[feat], self.ValidAcc[feat]])
            comb.Title = "Accuracy for Feature: " + feat
            comb.Filename = "Accuracy_" + feat
            comb.SaveFigure()

        merge = []
        for feat in self.TrainAcc:
            self.TrainAcc[feat].Title = feat
            self.TrainAcc[feat].Marker = None
            self.TrainAcc[feat].Color = None
            merge.append(self.TrainAcc[feat])
        
        self.OutputDirectory = Output + "/training/plots/accuracy"
        Comb = self.MergePlots(merge)
        Comb.Filename = "MergedFeatureTrainingAccuracy"
        Comb.Title = "Accuracy for All Features - Training"
        Comb.SaveFigure()

        loss, acc = [], []
        for node in self.NodeValidLoss:
            self.OutputDirectory = Output + "/training/plots/loss"
            
            # Continue here 
            self.MakeLossPlot(self.NodeValidLoss[node], node, "Validation", ".")
            self.MakeLossPlot(self.NodeTrainLoss[node], node, "Training", ",")
            loss.append(self.NodeValidLoss[node])
            loss.append(self.NodeTrainLoss[node])

            self.OutputDirectory = Output + "/training/plots/accuracy"
            self.MakeAccuracyPlot(self.NodeValidAcc[node], node, "Validation", ".")
            self.MakeAccuracyPlot(self.NodeTrainAcc[node], node, "Training", ",")
            acc.append(self.NodeValidAcc[node])
            acc.append(self.NodeTrainAcc[node])
            exit()

        Comb = self.MergePlots(loss)
        Comb.Filename = "NodeFeatureLoss"
        Comb.Title = "Feature "
        Comb.SaveFigure()







import time
class FigureContainer:

    def __init__(self):
        self.OutputDirectory = None
        self.training = Training()

    def AddEpoch(self, epoch, vals):
        if "training" in vals:
            self.training.AddEpoch(epoch, vals["training"])

    def Compile(self):
        self.training.Compile(self.OutputDirectory)









