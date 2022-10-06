from AnalysisTopGNN.Plotting.TemplateHistograms import TH1FStack
from AnalysisTopGNN.Plotting.TemplateLines import TLineStack
from Tooling import Tools



class Template(Tools):
    
    def __init__(self):
        pass

    def AddPoint(self, Src, Trgt, cat = False):
        for feat in Src:
            if feat not in Trgt:
                Trgt[feat] = []
            
            if cat == False:
                Trgt[feat].append(Src[feat])
            else:
                Trgt[feat] += Src[feat]
    
    def TemplateDict(self, OutputDirectory):
        Plots = {
                    "Tile" : "", 
                    "Filename" : "", 
                    "Style" : "ATLAS", 
                    "yTitle" : "", 
                    "xTitle" : "", 
                    "OutputDirectory" : OutputDirectory
                }
        return Plots 





class Training:
    
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
        self.AddPoint(dic["TrainingLoss"], self.TrainLoss)
        self.AddPoint(dic["TrainingAccuracy"], self.TrainAcc)
        self.AddPoint(dic["ValidationLoss"], self.ValidLoss)
        self.AddPoint(dic["ValidationAccuracy"], self.ValidAcc)
        self.Epochs.append(epoch)

        self.EpochTime.append(dic["EpochTime"])
        self.FoldTime += dic["FoldTime"]
        self.kFolds += dic["KFolds"]
        self.AddPoint(dic["NodeTime"], self.NodeTime, True) 

        self.AddPoint(dic["NodeTrainingLoss"], self.NodeTrainLoss)
        self.AddPoint(dic["NodeTrainingAccuracy"], self.NodeTrainAcc)
        self.AddPoint(dic["NodeValidationLoss"], self.NodeValidLoss)
        self.AddPoint(dic["NodeValidationAccuracy"], self.NodeValidAcc)

    
    def Compile(self):
        Config = self.TemplateDict("training/plots/loss")
        Config["Filename"] = "LossPlot"
        Config["xTitle"] = "Epoch"
        Config["yTitle"] = "Loss (a.u.)"

        for feat in self.TrainLoss:
            Config["Title"] = "Training"
            Config["yData"] = self.TrainLoss
            Config["xData"] = self.Epochs 


import time
class FigureContainer:

    def __init__(self, Epochs, Container):
        self.training = Training()

        for epoch in Epochs:
            for key in Container[epoch]:
                if key == "training":
                    self.training.AddEpoch(epoch, Container[epoch][key])
        
        self.training.Compile()









