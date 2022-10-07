from Tooling import Template
from AnalysisTopGNN.Plotting import TLine, TH1FStack

class Template(Template):

    def __init__(self):
        pass        

    def AddEpoch(self, epoch, dic):
        for feat in dic["AUC/AllCollected"]:
            if feat not in self.Acc:
                self.Acc[feat] = {}
                self.Loss[feat] = {}
                self.AUC[feat] = {}
            
            self.AUC[feat][epoch] = dic["AUC/AllCollected"][feat]
            self.ROC[epoch] = dic["ROC/CombinedFeatures"]

            self.Acc[feat][epoch] = dic["Accuracy/" + feat]
            self.Loss[feat][epoch] = dic["Loss/" + feat]

        self.AddEpochEdge(dic, epoch)
        self.AddEpochNode(dic, epoch)

    def Compile(self, Output, Mode = "all"):
        if len(self.Acc) == 0:
            self.Warning(Mode + " has no samples. Make the HDF5 sample larger. Skipping...")
            return 
        for i in self.Acc:
            self.OutputDirectory = Output + "/" + Mode + "/plots/"
            self.MakeAccuracyPlot(self.Acc, i, i, "-")

            self.OutputDirectory = Output + "/" + Mode + "/plots/"
            self.MakeLossPlot(self.Loss, i, i, "-")

        self.OutputDirectory = Output + "/" + Mode + "/plots/"
        comb = self.MergePlots([self.Acc[i] for i in self.Acc])
        comb.Title = "Accuracy for Graph Neural Features"
        comb.Filename = "AccuracyForAllFeatures"
        comb.SaveFigure()


        self.OutputDirectory = Output + "/" + Mode + "/plots/"
        comb = self.MergePlots([self.Loss[i] for i in self.Loss])
        comb.Title = "Loss for Graph Neural Features"
        comb.Filename = "LossForAllFeatures"
        comb.SaveFigure()
       
        self.SortEpoch(self.ROC)
        for ep in self.ROC:
            Aggre = []
            self.OutputDirectory = Output + "/" + Mode + "/plots/ROC-Epoch/"
            for feat in self.ROC[ep]:
                plot = self.TemplateROC(self.OutputDirectory, self.ROC[ep][feat]["FPR"], self.ROC[ep][feat]["TPR"])
                plot["Title"] = feat 
                Aggre.append(TLine(**plot))
            com = self.MergePlots(Aggre)
            com.Title = "ROC Curve for Epoch " + str(ep)
            com.Filename = "Epoch_" + str(ep)
            com.SaveFigure() 
        
        for feat in self.AUC:
            self.SortEpoch(self.AUC[feat])
            plot = self.TemplateTLine(Output + "/" + Mode + "/plots")
            plot["Title"] = feat
            plot["xTitle"] = "Epoch"
            plot["yTitle"] = "Area Under ROC Curve"
            plot["xMin"] = 0
            plot["yMin"] = 0 
            plot["xMax"] = 1
            plot["yMax"] = 1
            plot["xData"] = list(self.AUC[feat])
            plot["yData"] = self.UnNestList(list(self.AUC[feat].values()))
            self.AUC[feat] = TLine(**plot)
        
        self.OutputDirectory = Output + "/" + Mode + "/plots"
        com = self.MergePlots(list(self.AUC.values()))
        com.Title = "Area under ROC Curve for All Features with respect to Epoch"
        com.Filename = "AUC_AllFeatures"
        com.SaveFigure()
        
        out = Output + "/" + Mode + "/plots/EdgeMass-Epoch"
        self.MakeMassPlot(self.EdgeMass, "Edge", out)
        out = Output + "/" + Mode + "/plots/NodeMass-Epoch"
        self.MakeMassPlot(self.NodeMass, "Node", out)
        
        self.OutputDirectory = Output + "/" + Mode + "/plots/ProcessReconstruction-Edge"
        self.MakeReconstructionProcessEfficiency(self.EdgeMassPrcEff)       
        self.MakeReconstructionEfficiency(self.EdgeMassAll) 

        self.OutputDirectory = Output + "/" + Mode + "/plots/ProcessReconstruction-Node"
        self.MakeReconstructionProcessEfficiency(self.NodeMassPrcEff)      
        self.MakeReconstructionEfficiency(self.NodeMassAll)

class Train(Template):

    def __init__(self):
        self.EdgeMass = {}
        self.NodeMass = {}

        self.EdgeMassPrcCompo = {}
        self.NodeMassPrcCompo = {}

        self.EdgeMassPrcEff = {}
        self.NodeMassPrcEff = {}

        self.EdgeMassAll = {}
        self.NodeMassAll = {}

        self.ROC = {}
        self.AUC = {}

        self.Loss = {}
        self.Acc = {} 
        self.VerboseLevel = 1
        self.Caller = "Train"


class Test(Template):

    def __init__(self):
        self.EdgeMass = {}
        self.NodeMass = {}

        self.EdgeMassPrcCompo = {}
        self.NodeMassPrcCompo = {}

        self.EdgeMassPrcEff = {}
        self.NodeMassPrcEff = {}

        self.EdgeMassAll = {}
        self.NodeMassAll = {}

        self.ROC = {}
        self.AUC = {}

        self.Loss = {}
        self.Acc = {}
        self.VerboseLevel = 1
        self.Caller = "Test"

class All(Template): 

    def __init__(self):
        self.EdgeMass = {}
        self.NodeMass = {}

        self.EdgeMassPrcCompo = {}
        self.NodeMassPrcCompo = {}

        self.EdgeMassPrcEff = {}
        self.NodeMassPrcEff = {}

        self.EdgeMassAll = {}
        self.NodeMassAll = {}

        self.ROC = {}
        self.AUC = {}

        self.Loss = {}
        self.Acc = {}
        self.VerboseLevel = 1
        self.Caller = "All"
