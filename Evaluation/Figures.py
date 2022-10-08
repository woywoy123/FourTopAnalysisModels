from TrainingPlots import Training
from AllPlots import All, Train, Test
from Tooling import Template
from LogDump import LogDumper
   
class FigureContainer:

    def __init__(self):
        self.OutputDirectory = None
        self.training = Training()
        self.all = All()
        self.test = Test()
        self.train = Train()
        self.TrainingPlots = {}
        self.AllPlots = {}

    def AddEpoch(self, epoch, vals):
        if "training" in vals:
            self.training.AddEpoch(epoch, vals["training"])
        elif "all" in vals:
            self.all.AddEpoch(epoch, vals["all"])
        elif "train" in vals:
            self.train.AddEpoch(epoch, vals["train"])
        elif "test" in vals:
            self.test.AddEpoch(epoch, vals["test"])

    def Compile(self):
        self.TrainingPlots |= self.training.Compile(self.OutputDirectory)
        self.AllPlots |= self.all.Compile(self.OutputDirectory, "all")
        #self.test.Compile(self.OutputDirectory, "test")
        #self.train.Compile(self.OutputDirectory, "train")
           

class ModelComparison(Template, LogDumper):

    def __init__(self):
        self.TrainingAccuracy = {}
        self.EpochTime = {}

        # ==== Plots for all samples ==== #
        self.AllAccuracy = {}
        self.AllLoss = {}
        self.AllAUC = {}
        self.AllEdgeEffPrc = {}
        self.AllEdgeEff = {}
        self.AllNodeEffPrc = {}
        self.AllNodeEff = {}

        # ==== Plots for test samples ==== #
        self.AllAccuracy = {}
        self.AllLoss = {}
        self.AllAUC = {}
        self.AllEdgeEffPrc = {}
        self.AllEdgeEff = {}
        self.AllNodeEffPrc = {}
        self.AllNodeEff = {}
 
        # ==== Plots for train samples ==== #
        self.AllAccuracy = {}
        self.AllLoss = {}
        self.AllAUC = {}
        self.AllEdgeEffPrc = {}
        self.AllEdgeEff = {}
        self.AllNodeEffPrc = {}
        self.AllNodeEff = {}
 


        self.OutputDirectory = None
        self.Colors = {}
        self._S = " | "

    def AddModel(self, name, Model):
        self.TrainingAccuracy[name] = Model.Figure.TrainingPlots["Accuracy"]
        self.EpochTime[name] = Model.Figure.TrainingPlots["EpochTime"]

        self.AllAccuracy[name] = Model.Figure.AllPlots["Accuracy"]
        self.AllLoss[name] = Model.Figure.AllPlots["Loss"]
        self.AllAUC[name] = Model.Figure.AllPlots["AUC"]

        self.AllEdgeEffPrc[name] = Model.Figure.AllPlots["EdgeProcessEfficiency"]
        self.AllEdgeEff[name] = Model.Figure.AllPlots["EdgeEfficiency"]

        self.AllNodeEffPrc[name] = Model.Figure.AllPlots["NodeProcessEfficiency"]
        self.AllNodeEff[name] = Model.Figure.AllPlots["NodeEfficiency"]

    def Compare(self, dic, Title, xTitle, yTitle, yMax, Filename):
        for i in dic:
            dic[i].Title = i
        com = self.MergePlots(list(dic.values()), self.OutputDirectory)
        com.xTitle = xTitle
        com.yTitle = yTitle
        com.yMin = -0.1
        com.yMax = yMax
        com.Filename = Filename
        com.Title = Title
        com.SaveFigure()
        out = self.DumpTLines(com.Lines)
        self.WriteText(out, self.OutputDirectory + "/" + Filename)
        return com

    def CompareEpochTime(self):
        
        com = self.Compare(self.EpochTime, "Time Elapsed at each Epoch", "Epoch", "Time (s)", None, "EpochTime")
        for l in com.Lines:
            self.Colors[l.Title] = l.Color

    def Organize(self, dic):
        lines = list(dic.values())
        names = list(dic)
        
        Features = {}
        for name, l in zip(names, lines):
            Lines = list(l.values()) if isinstance(l, dict) else l.Lines
            for line in Lines:
                feat = line.Title
                if feat not in Features:
                    Features[feat] = {}
                Features[feat][name] = line
                line.Color = self.Colors[name]
        return Features 

    def CompareAccuracy(self, dic, prefix):
        Features = self.Organize(dic)
        for feat in Features:
            self.Compare(Features[feat], "Accuracy of Feature " + feat + " Prediction", "Epoch", "Accuracy (%)", 101, prefix + "-" + feat)

    def CompareLoss(self, dic, prefix):
        Features = self.Organize(dic)
        for feat in Features:
            self.Compare(Features[feat], "Loss from Predicting " + feat + " Prediction", "Epoch", "Loss (a.u)", None, prefix + "-" + feat)

    def CompareAUC(self, dic, prefix):
        Features = self.Organize(dic)
        for feat in Features:
            self.Compare(Features[feat], "Achieved Area under ROC Curve for Feature " + feat, "Epoch", "AUC (Higher is Better)", 1.1, prefix + "-" + feat)

    def CompareReco(self, dic, prefix, prc = ""):
        Features = self.Organize(dic)
        for feat in Features:
            self.Compare(Features[feat], "Top Reconstruction Efficiency of Feature " + feat + " Prediction" + prc, "Epoch", "Reconstruction Efficiency (%)", 101, prefix + "-" + feat)
    
    def CompareRecoByProcess(self, dic, prefix):
        Dic = {} 
        for model in dic:
            for feat in dic[model]:
                for p in dic[model][feat].Lines:
                    if p.Title not in Dic:
                        Dic[p.Title] = {}
                    if model not in Dic[p.Title]:
                        Dic[p.Title][model] = {}
                    Dic[p.Title][model][feat] = p
                    p.Title = feat
        tmp = self.OutputDirectory
        for prc in Dic:
            self.OutputDirectory = tmp + "/" + prc
            self.CompareReco(Dic[prc], prefix, " (" + prc +")")

    def Compile(self):
        RootDir = self.OutputDirectory + "/ModelComparison"
        AccDir = RootDir + "/accuracy"
        LossDir = RootDir + "/loss"
        AUCDir = RootDir + "/auc"
        RecoEff = RootDir + "/reco"

        self.OutputDirectory = RootDir
        self.CompareEpochTime() 

        self.OutputDirectory = AccDir
        self.CompareAccuracy(self.TrainingAccuracy, "training")
        self.CompareAccuracy(self.AllAccuracy, "all")
            
        self.OutputDirectory = LossDir 
        self.CompareLoss(self.AllLoss, "all")

        self.OutputDirectory = AUCDir
        self.CompareAUC(self.AllAUC, "all")

        self.OutputDirectory = RecoEff
        self.CompareReco(self.AllEdgeEff, "edge-all")
        self.CompareReco(self.AllNodeEff, "node-all")

        self.CompareRecoByProcess(self.AllEdgeEffPrc, "edge-all")
        self.CompareRecoByProcess(self.AllNodeEffPrc, "node-all")


