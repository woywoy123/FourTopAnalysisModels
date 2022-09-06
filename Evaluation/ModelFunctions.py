from AnalysisTopGNN.IO import UnpickleObject, Directories
from AnalysisTopGNN.Tools import Threading
from AnalysisTopGNN.Plotting import *
import statistics
import math

class Evaluation(Directories):

    def __init__(self, Source, Model):
        self.VerboseLevel = 0
        self.Verbose = True
        self.Caller = "EVALUATION"

        self.ModelDirectoryName = Model 
        self.SourceDirectory = Source
        self._Epochs = {}
        self._Stats = None
        self.Threads = 6

        self._SampleInformation = {}
        self._TrainingInformation = {}
        self._ModelTrainingParameters = {}
        
        self._TrainingAccuracy = {}
        self._TrainingLoss = {}

        self._ValidationAccuracy = {}
        self._ValidationLoss = {}

        self._SkippedNodes = {}

        self._ModelKeys = None

    def ReadStatistics(self):
        def function(F, out = []):
            for i in F:
                out.append(UnpickleObject(i))
            return out

        def MakeDict(out = {}):
            for i in self._Epochs:
                out[i] = {}
                for key in self._ModelKeys:
                    out[i][key] = {}
                    out[i][key]["_Merged"] = []
                    
                    for node in self._nNodes:
                        out[i][key][node] = []
            return out
        
        for i in self.ListFilesInDir(self.SourceDirectory + "/" + self.ModelDirectoryName + "/Statistics/"):
            epoch = i.split('_')[-1].replace('.pkl', '')
    
            if epoch == "Done":
                self._Stats = i
            else:
                self._Epochs[int(epoch)] = i

        TH = Threading(list(self._Epochs.values()), function, threads = self.Threads)
        TH.Start()
        
        for i in self._Epochs:
            self._Epochs[i] = TH._lists[i-1]
        self._Stats = UnpickleObject(self._Stats)
        
        for i in ["Samples", "Tree", "Level", "SelfLoop", "Start", "End"]:
            self._SampleInformation[i] = self._Stats[i]
        self._ModelTrainingParameters["BatchSize"] = self._Stats["BatchSize"]
        self._ModelTrainingParameters |= self._Stats["Model"]

        self._ModelKeys = list(self._Stats["Training_Accuracy"])


        self._nNodes = list(set([i for t in self._Stats["n_Node_Files"] for i in t]))

        self._TrainingInformation["EpochTime"] = []
        self._TrainingInformation["AverageFoldTime"] = []
        self._TrainingInformation["AverageNodeFoldTime"] = {}
        for i in self._nNodes:
            self._TrainingInformation["AverageNodeFoldTime"][i] = []

        self._TrainingAccuracy |= MakeDict()
        self._ValidationAccuracy |= MakeDict()
        
        self._TrainingLoss |= MakeDict()
        self._ValidationLoss |= MakeDict()

    def ProcessEpoch(self, epoch):
        self._TrainingInformation["EpochTime"] += self._Epochs[epoch]["EpochTime"]
        
        tme = []
        for i in range(len(self._Epochs[epoch]["kFold"])):
            ep = self._Epochs[epoch]
            kFT = ep["FoldTime"][i]
            node = ep["Nodes"][i]
            
            self._TrainingInformation["AverageNodeFoldTime"][node] += kFT 
            tme += kFT
        self._TrainingInformation["AverageFoldTime"].append(tme)

        for key in self._ModelKeys:
            ev = self._Epochs[epoch]

            tr_a = ev["Training_Accuracy"][key]
            tr_l = ev["Training_Loss"][key]

            va_a = ev["Validation_Accuracy"][key]
            va_l = ev["Validation_Loss"][key]
 
            for n in range(len(ev["Nodes"])):
                node = ev["Nodes"][n]
                
                for j in range(len(ev["kFold"][n])):
                    indx = n*len(ev["kFold"][n])+j
                    
                    self._TrainingAccuracy[epoch][key]["_Merged"] += tr_a[indx]
                    self._TrainingAccuracy[epoch][key][node] += tr_a[indx]

                    self._ValidationAccuracy[epoch][key]["_Merged"] += va_a[indx]
                    self._ValidationAccuracy[epoch][key][node] += va_a[indx]

                    self._TrainingLoss[epoch][key]["_Merged"] += tr_l[indx]
                    self._TrainingLoss[epoch][key][node] += tr_l[indx]

                    self._ValidationLoss[epoch][key]["_Merged"] += va_l[indx]
                    self._ValidationLoss[epoch][key][node] += va_l[indx]
    
    def EpochLoop(self):
        def MakeStats(inpt, node = None):
            if len(inpt) == 0 and node != None:
                self._SkippedNodes[node] = []
            if len(inpt) == 0:
                return [None, None, None, None] 

            mean = statistics.mean(inpt)
            stdErr = statistics.pstdev(inpt, mean)/math.sqrt(len(inpt))
            return [mean, stdErr, stdErr, inpt]

        for i in range(len(self._Epochs)):
            self.ProcessEpoch(i+1)
        
        for node in self._nNodes:
            self._TrainingInformation["AverageNodeFoldTime"][node] = MakeStats(self._TrainingInformation["AverageNodeFoldTime"][node], node)

        for epoch in range(len(self._Epochs)):
            self._TrainingInformation["AverageFoldTime"][epoch] = MakeStats(self._TrainingInformation["AverageFoldTime"][epoch])        
            
            for key in self._ModelKeys:
                tr_a = self._TrainingAccuracy[epoch+1][key]
                va_a = self._ValidationAccuracy[epoch+1][key]
                tr_l = self._TrainingLoss[epoch+1][key]
                va_l = self._ValidationLoss[epoch+1][key]

                self._TrainingAccuracy[epoch+1][key]["_Merged"] = MakeStats(tr_a["_Merged"])
                self._ValidationAccuracy[epoch+1][key]["_Merged"] = MakeStats(va_a["_Merged"])
                self._TrainingLoss[epoch+1][key]["_Merged"] = MakeStats(tr_l["_Merged"])
                self._ValidationLoss[epoch+1][key]["_Merged"] = MakeStats(va_l["_Merged"])

                for node in self._nNodes:
                    self._TrainingAccuracy[epoch+1][key][node] = MakeStats(tr_a[node])
                    self._ValidationAccuracy[epoch+1][key][node] = MakeStats(va_a[node])
                    self._TrainingLoss[epoch+1][key][node] = MakeStats(tr_l[node])
                    self._ValidationLoss[epoch+1][key][node] = MakeStats(va_l[node])
    
    def MakePlots(self, OutputDir):
        def CompilerLineError(lst, xvar, Title = None, xTitle = None, yTitle = None, Filename = None, Outdir = None):
            y = [lst[i][0] for i in xvar if lst[i][0] != None]
            up_y = [lst[i][1] for i in xvar if lst[i][1] != None]
            down_y = [lst[i][2] for i in xvar if lst[i][2] != None]
            x = [i for i in xvar if lst[i][0] != None]
            
            line = TLine(xData = x, yData = y, 
                    up_yData = up_y, down_yData = down_y, 
                    Title = Title, Filename = Filename, OutputDirectory = Outdir,
                    xTitle = xTitle, yTitle = yTitle, Style = "ATLAS")
            return [line]

        def CompilerLine(lst, xvar, Title = None, xTitle = None, yTitle = None, Filename = None, Outdir = None):
            y = [lst[i][0] for i in xvar if lst[i][0] != None]
            x = [i for i in xvar if lst[i][0] != None]
            
            line = TLine(xData = x, yData = y, 
                    up_yData = up_y, down_yData = down_y, 
                    Title = Title, Filename = Filename, OutputDirectory = Outdir,
                    xTitle = xTitle, yTitle = yTitle, Style = "ATLAS")
            return [line]

        def CompilerHist(lst, xvar, Title = None, xTitle = None, yTitle = None, Filename = None, Outdir = None):
            hists = [TH1F(xData = lst[i][3], Title = Title + " - " + str(i), xTitle = xTitle, yTitle = yTitle, 
                    Filename = Filename + "-" +str(i), OutputDirectory = Outdir + "/Histograms", Style = "ATLAS", xBins = 100) for i in xvar if lst[i][3] != None]
            return hists



        self.Lines = []
        self.Hists = []
        self.Lines += CompilerLineError(self._TrainingInformation["AverageNodeFoldTime"], self._nNodes, 
                                "Average Node Folding Time", "Number of Nodes", "Time (s)", 
                                "AverageNodeFoldTime", OutputDir + "/" + self.ModelDirectoryName + "/Timing")

        self.Hists += CompilerHist(self._TrainingInformation["AverageNodeFoldTime"], self._nNodes, 
                                "Average Folding Time for Node", "Time (s)", "Entries", 
                                "AverageNodeFoldTime", OutputDir + "/" + self.ModelDirectoryName + "/Timing")

        self.Lines += CompilerLineError(self._TrainingInformation["AverageFoldTime"], self._nNodes, 
                                "Average Folding Time", "Epoch ", "Time (s)", 
                                "AverageFoldTime", OutputDir + "/" + self.ModelDirectoryName + "/Timing")

        self.Hists += CompilerHist(self._TrainingInformation["AverageFoldTime"], self._nNodes, 
                                "Average Folding Time", "Time (s)", "Entries", 
                                "AverageFoldTime", OutputDir + "/" + self.ModelDirectoryName + "/Timing")
        

        for key in self._ModelKeys:
    

            Label = "Training"
            inpt = {epoch : self._TrainingAccuracy[epoch][key]["_Merged"] for epoch in self._Epochs}
            self.Lines += CompilerLineError(inpt, self._Epochs, Label + " Accuracy of " + key, "Epoch ", "Accuracy", 
                                    "All_"+key, OutputDir + "/" + self.ModelDirectoryName + "/" + key)

            self.Hists += CompilerHist(inpt, self._Epochs, Label + " Accuracy of " + key, "Accuracy", "Entries", 
                                "All_"+key, OutputDir + "/" + self.ModelDirectoryName + "/" + key)



            Label = "Validation"
            inpt = {epoch : self._ValidationAccuracy[epoch][key]["_Merged"] for epoch in self._Epochs}
            self.Lines += CompilerLineError(inpt, self._Epochs, Label + " Accuracy of " + key, "Epoch ", "Accuracy", 
                                    "All_"+key, OutputDir + "/" + self.ModelDirectoryName + "/" + key)

            self.Hists += CompilerHist(inpt, self._Epochs, Label + " Accuracy of " + key, "Accuracy", "Entries", 
                                "All_"+key, OutputDir + "/" + self.ModelDirectoryName + "/" + key)
 

            continue


            #self._ValidationAccuracy[epoch+1][key]["_Merged"]
            #self._TrainingLoss[epoch+1][key]["_Merged"]
            #self._ValidationLoss[epoch+1][key]["_Merged"]

            #for node in self._nNodes:
            #    self._TrainingAccuracy[epoch+1][key][node] = MakeStats(tr_a[node])
            #    self._ValidationAccuracy[epoch+1][key][node] = MakeStats(va_a[node])
            #    self._TrainingLoss[epoch+1][key][node] = MakeStats(tr_l[node])
            #    self._ValidationLoss[epoch+1][key][node] = MakeStats(va_l[node])

    def Compile(self):
        def function(inpt):
            for i in inpt:
                i.VerboseLevel = 0
                i.SaveFigure()
                del i
            return [0 for i in inpt]
        
        TH = Threading(self.Hists, function)
        TH.Start()

        TH = Threading(self.Lines, function)
        TH.Start()

