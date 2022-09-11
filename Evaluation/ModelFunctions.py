from AnalysisTopGNN.IO import UnpickleObject, Directories, WriteDirectory
from AnalysisTopGNN.Tools import Threading
from AnalysisTopGNN.Plotting import *
from AnalysisTopGNN.Reconstruction import Reconstructor
import statistics
import math
import torch

class Evaluation(Directories, WriteDirectory):

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
        def CompilerLineError(lst, xvar, Title = None, xTitle = None, yTitle = None, Filename = None, Outdir = None, yMin = 0, yMax = 1, Log = False):
            y = [lst[i][0] for i in xvar if lst[i][0] != None]
            up_y = [lst[i][1] for i in xvar if lst[i][1] != None]
            down_y = [lst[i][2] for i in xvar if lst[i][2] != None]
            x = [i for i in xvar if lst[i][0] != None]
            
            if len(y) == 0:
                return []

            line = TLine(xData = x, yData = y, yMin = yMin, yMax = yMax,
                    up_yData = up_y, down_yData = down_y, 
                    Title = Title, Filename = Filename, OutputDirectory = Outdir,
                    xTitle = xTitle, yTitle = yTitle, Style = "ATLAS", Logarithmic  = Log)
            return [line]

        def MergeLines(Lines, Title = None, xTitle = None, yTitle = None, Filename = None, Outdir = None, yMin = 0, yMax = 1, Log = False):
            for i in Lines:
                if "Training" in i.Title:
                    i.Marker = "."
                    i.LineStyle = "-"
                    i.Label = i.Title
                elif "Validation" in i.Title:
                    i.Marker = "."
                    i.LineStyle = ":"
                    i.Label = i.Title

            if len(Lines) == 0:
                return []

            line = CombineTLine(Lines = Lines, Title = Title, xTitle = xTitle, yTitle = yTitle, 
                    Filename = Filename, OutputDirectory = Outdir,Style = "ATLAS", 
                    yMin = yMin, yMax = yMax, Logarithmic = Log)
            return [line]

        def CompilerHist(lst, xvar, Title = None, xTitle = None, yTitle = None, Filename = None, Outdir = None, xMin = 0, yMin = 0, xMax = 1):
            hists = [TH1F(xData = lst[i][3], Title = str(Title), xTitle = xTitle, yTitle = yTitle, xMin = xMin, yMin = yMin, xMax = xMax,
                    Filename = Filename, OutputDirectory = Outdir, Style = "ATLAS", xBins = 100) for i in xvar if lst[i][3] != None]
            return hists

        def CompilerCombinedHist(lst, xvar, Title = None, xTitle = None, yTitle = None, Filename = None, Outdir = None, xMin = 0, yMin = 0, xMax = 1):
            hists = [CombineTH1F(Histograms = lst[i], Title = str(Title), xTitle = xTitle, yTitle = yTitle, xMin = xMin, yMin = yMin, xMax = xMax,
                    Filename = Filename + "-" +str(i), OutputDirectory = Outdir, Style = "ATLAS", xBins = 100) for i in xvar]
            return hists

        def function(inpt):
            for i in inpt:
                i.VerboseLevel = 0
                i.SaveFigure()
                del i
            return [0 for i in inpt]


        self.Lines = []
        self.Hists = []
        
        Out_Dir = OutputDir + "/" + self.ModelDirectoryName

        # Timing Information of the Training and Validation Time 
        self.Lines += CompilerLineError(self._TrainingInformation["AverageNodeFoldTime"], self._nNodes, 
                                "Average Node Folding Time", "Number of Nodes", "Time (s)", 
                                "AverageNodeFoldTime", Out_Dir + "/Time", yMax = None)

        self.Hists += CompilerHist(self._TrainingInformation["AverageNodeFoldTime"], self._nNodes, 
                                "Average Folding Time for Node", "Time (s)", "Entries", 
                                "node_", Out_Dir + "/Time/AverageNodeFoldTimeHistograms", xMax = None)

        self.Lines += CompilerLineError(self._TrainingInformation["AverageFoldTime"], [epoch -1 for epoch in self._Epochs], 
                                "Average Folding Time After each Epoch", "Epoch ", "Time (s)", 
                                "AverageFoldTime",  Out_Dir + "/Time", yMin = None, yMax = None)

        self.Hists += CompilerHist(self._TrainingInformation["AverageFoldTime"], [epoch -1 for epoch in self._Epochs], 
                                "Average Folding Time After each Epoch", "Time (s)", "Entries", 
                                "epoch_", Out_Dir + "/Time/AverageFoldTimeHistograms", xMax = None)

        Collect_TA = []
        Collect_VA = []
        
        Collect_TL = []
        Collect_VL = []
        for key in self._ModelKeys:
            
            AccHist = []
            AccLine = []
            tmp = {epoch : self._TrainingAccuracy[epoch][key]["_Merged"] for epoch in self._Epochs}
            AccLine += CompilerLineError(tmp, list(self._Epochs), "Training", "Epoch", "Accuracy")
            AccHist += CompilerHist(tmp, list(self._Epochs), "Training", "Accuracy", "Entries")
            Collect_TA += CompilerLineError(tmp, list(self._Epochs), key, "Epoch", "Accuracy")
            
            tmp = {epoch : self._ValidationAccuracy[epoch][key]["_Merged"] for epoch in self._Epochs}
            AccLine += CompilerLineError(tmp, list(self._Epochs), "Validation", "Epoch", "Accuracy")
            AccHist += CompilerHist(tmp, list(self._Epochs), "Validation", "Accuracy", "Entries")
            Collect_VA += CompilerLineError(tmp, list(self._Epochs), key, "Epoch", "Accuracy")
            
            self.Lines += MergeLines(AccLine, "Training/Validation Accuracy Feature: " + key, "Epoch", "Accuracy", "Accuracy", Out_Dir + "/" + key)
            self.Hists += CompilerCombinedHist([[AccHist[l], AccHist[int(len(AccHist)/2) + l]] for l in range(int(len(AccHist)/2))], 
                    [epoch -1 for epoch in self._Epochs], "Training/Validation Accuracy Feature: " + key, "Accuracy", "Entries", 
                    "Epoch", Out_Dir + "/" + key + "/Histogram")



            LossLine = []
            tmp = {epoch : self._TrainingLoss[epoch][key]["_Merged"] for epoch in self._Epochs}
            LossLine += CompilerLineError(tmp, list(self._Epochs), "Training", "Epoch", "Loss")
            Collect_TL += CompilerLineError(tmp, list(self._Epochs), key, "Epoch", "Loss")

            tmp = {epoch : self._ValidationLoss[epoch][key]["_Merged"] for epoch in self._Epochs}
            LossLine += CompilerLineError(tmp, list(self._Epochs), "Validation", "Epoch", "Loss")
            Collect_VL += CompilerLineError(tmp, list(self._Epochs), key, "Epoch", "Loss")

            self.Lines += MergeLines(LossLine, "Training/Validation Loss Feature: " + key, "Epoch", "Loss", 
                    "Loss", Out_Dir + "/" + key, yMin = None, yMax = None, Log = True)

            for node in self._nNodes:

                Node_Collect_Accuracy = []

                tmp = {epoch : self._TrainingAccuracy[epoch][key][node] for epoch in self._Epochs}
                Node_Collect_Accuracy += CompilerLineError(tmp, list(self._Epochs), "Training", "Epoch", "Accuracy")

                tmp = {epoch : self._ValidationAccuracy[epoch][key][node] for epoch in self._Epochs}
                Node_Collect_Accuracy += CompilerLineError(tmp, list(self._Epochs), "Validation", "Epoch", "Accuracy")

                self.Lines += MergeLines(Node_Collect_Accuracy, "Training/Validation Accuracy Node: " + str(node), "Epoch", "Accuracy", 
                        "Accuracy_Node-"+str(node), Out_Dir + "/" +  key + "/NodeAccuracy")


                Node_Collect_Loss = []

                tmp = {epoch : self._TrainingLoss[epoch][key][node] for epoch in self._Epochs}
                Node_Collect_Loss += CompilerLineError(tmp, list(self._Epochs), "Training", "Epoch", "Loss")

                tmp = {epoch : self._ValidationLoss[epoch][key][node] for epoch in self._Epochs}
                Node_Collect_Loss += CompilerLineError(tmp, list(self._Epochs), "Validation", "Epoch", "Loss")

                self.Lines += MergeLines(Node_Collect_Loss, "Training/Validation Loss Node: " + str(node), "Epoch", "Loss", 
                        "Loss_Node-"+str(node), Out_Dir + "/" +  key + "/NodeLoss")

        
        # Generate Similar Color Profiles
        CombineTLine(Lines = Collect_TA).ConsistencyCheck()
        CombineTLine(Lines = Collect_VA).ConsistencyCheck()

        CombineTLine(Lines = Collect_TL).ConsistencyCheck()
        CombineTLine(Lines = Collect_VL).ConsistencyCheck()

        for i in range(len(Collect_TL)):
            Collect_TL[i].LineStyle = "-"
            Collect_TL[i].Marker = "."
            Collect_TL[i].Label = Collect_TL[i].Title
            
            Collect_TA[i].LineStyle = "-"
            Collect_TA[i].Marker = "."
            Collect_TA[i].Label = Collect_TA[i].Title

            Collect_VL[i].LineStyle = ":"
            Collect_VL[i].Marker = "."
            Collect_VA[i].LineStyle = ":"
            Collect_VA[i].Marker = "."

        self.Lines += MergeLines(Collect_TA + Collect_VA, "Accuracy of Training (Solid) / Validation (Dotted) \n All Features", "Epoch", "Accuracy", 
                "Accuracy", Out_Dir + "/Summary")
        self.Lines += MergeLines(Collect_TL + Collect_VL, "Loss of Training (Solid) / Validation (Dotted) \n All Features", "Epoch", "Loss", 
                "Loss", Out_Dir + "/Summary", yMin = None, yMax = None, Log = True)

        TH = Threading(self.Hists, function, self.Threads)
        TH.Start()
            
        TH = Threading(self.Lines, function, self.Threads)
        TH.Start()
    
    def MakeLog(self, OutputDir):
        def Spacer(inpt, spacer):
            return str(inpt) + " "*(sp - len(str(inpt))) + " | "

        sp = 4
        self.pwd = OutputDir + "/" + self.ModelDirectoryName
        self._Table = {}
        self._Table |= {key : ["#" + "="*10 + " Training | Validation " + key + " Details " + "="*10 + "#"] for key in self._ModelKeys} 
        
        for i in self._Table:
            TA, VA, ep = [], [], []
            for epoch in self._Epochs:
                TA.append(self._TrainingAccuracy[epoch][i]["_Merged"][0])
                VA.append(self._ValidationAccuracy[epoch][i]["_Merged"][0])
                ep.append(epoch)
            
            self._Table[i].append("-"*sp + " Accuracy Performance Summary " + "-"*sp)
            self._Table[i].append("Best Training EPOCH: " + str(ep[TA.index(max(TA))]) + " | " + str(round(max(TA), 4)))
            self._Table[i].append("Worst Training EPOCH: " + str(ep[TA.index(min(TA))]) + " | " + str(round(min(TA), 4)))
            self._Table[i].append("Best Validation EPOCH: " + str(ep[VA.index(max(VA))]) + " | " + str(round(max(VA), 4)))
            self._Table[i].append("Worst Validation EPOCH: " + str(ep[VA.index(min(VA))]) + " | " + str(round(min(VA), 4)))
            self._Table[i].append("-"*sp*10 + "\n")

        self._Table["Time"] = ["#" + "="*10 + " Training Time Details " + "="*10 + "#"]
        
        for epoch in self._Epochs:
            st = "EPOCH: " + Spacer(epoch, sp)
            Timer = round(self._TrainingInformation["EpochTime"][epoch -1], 4)

            self._Table["Time"].append(st + Spacer(Timer, sp))
            for key in self._ModelKeys:
                Mean_T = round(self._TrainingAccuracy[epoch][key]["_Merged"][0], sp)
                Mean_V = round(self._ValidationAccuracy[epoch][key]["_Merged"][0], sp)
                self._Table[key].append(st + Spacer(Mean_T, sp*2) + Spacer(Mean_V, sp*2))
       
        for i in self._Table:
            self.WriteTextFile(self._Table[i], "/Summary", i)
        
        Model = ["="*sp + " Optimizer and Model Settings Summary " + "="*sp]
        Model.append("___ RUN NAME: " + self.ModelDirectoryName + "___")
        for i in self._ModelTrainingParameters:
            Model.append("-> " + str(i) + ": " + str(self._ModelTrainingParameters[i]))

        Model.append("_" * sp*10)

        Model.append("_"*sp + " Sample Details " + "_"*sp)
        s = self._SampleInformation
        for i in range(len(s["Samples"])):
            Model += ["".join([key + " | " + str(s[key][i]) + " | " for key in s])]

        self.WriteTextFile(Model, "/Summary", "RunSummary") 
   

class TorchScriptModel(Notification):

    def __init__(self, ModelPath, Device = "cpu", **kargs):
        self.Verbose = True
        self.VerboseLevel = 3
        self.Caller = "TorchScriptModel"
        self.ModelPath = ModelPath
        self.key_o = {}
        self.key_l = {}
        self.key_c = {}
        self._it = 0

        key = { str(l) : "" for l in open(ModelPath, "rb").readlines() if "/extra/" in str(l)}
        out = []
        for k in list(set([l.split("/")[-1] for t in key for l in t.split("\\") if "/extra/" in l])):
            x = "FB".join(k.split("FB")[:-1])
            if x != "":
                out.append(x)
            x = "PK".join(k.split("PK")[:-1])
            if x != "":
                out.append(x)
        
        key = {k : "" for k in list(set(out))}
        self._model = torch.jit.load(self.ModelPath, _extra_files = key)

        self.inpt_keys = [k.replace(")", "").replace(",", "") for k in str(self._model.forward.schema).split(" ") if "Tensor" not in k and "->" not in k][1:]
        self.forward() 

        outputs = []
        for i in self._model.graph.outputs():
            line = str(i).replace("\n", "").split("=")[1]
            line = ["%" + k.replace(")", "").replace(",", "").replace(" ", "") for k in line.split("%") if "::" not in k]
            outputs += line
            self._it += 1    

        if isinstance(kargs, dict):
            for i in list(kargs.values())[0]:
                self.GetNodeAsOutput(**i)
            return 

        self.key_o |= { k : key[k] for k in key if k.startswith("O_") }
        self.key_l |= { k.lstrip("L_") : key[k] for k in key if k.startswith("L_") }
        self.key_c |= { k.lstrip("C_") : key[k] for k in key if k.startswith("C_") }

        self.key_o |= {k : [outputs[j], None] for k, j in zip(self.key_o, range(len(outputs)))}
        
        if len(outputs) == len(key):
            return  

        self.Warning("OUTPUT NUMBERS DO NOT MATCH FOUND KEYS! THE FOLLOWING NODE MAPPING HAS BEEN PRODUCED:")
        for i in self.key_o:
            if isinstance(self.key_o[i], list):
                self.Warning("KEY: " + i + " NODE: " + str(self.key_o[i][0]))
            else:
                self.Warning("KEY MISSING: " + i)
                self.key_o[i] = [-1, None]
        
        self.Warning(">--------- Possible Nodes ---------<")
        for i in self._model.graph.nodes():
            if "NoneType" not in str(i):
                continue 
            n = str(i).replace("\n", "").split("=")
            self.Warning("Name: " + n[0] + " | Operation: " + "=".join(n[1:]))
        self.Warning(">----------------------------------<")

    def forward(self):
        self.forward = self
        self.forward.__code__ = self
        self.forward.__code__.co_varnames = self.inpt_keys
        self.forward.__code__.co_argcount = len(self.inpt_keys)

    def ShowNodes(self):
        self.Notify("------------ BEGIN GRAPH NODES -----------------")
        for i in self._model.graph.nodes():
            n = str(i).replace("\n", "").split(" : ")
            f = n[1].split("#")
            string = "Name: " + n[0] + " | Operation: " + f[0] 
            if len(f) > 1:
                string += " | Function Call: " + f[1].split("/")[-1]
            self.Notify(string)
        self.Notify("------------ END GRAPH NODES -----------------")

    def GetNodeAsOutput(self, name, node, loss, classification):
        if isinstance(name, str):
            names= {"O_" + name : node}
            self.key_o |= names
        else:
            return self.Warning("INVALID NAMES GIVEN!")
        
        for i in self._model.graph.nodes():
            n_ref = str(i).split(" = ")[0].split(":")[0].replace(" ", "")
            if n_ref != node:
                continue
            self.key_o["O_" + name] = [n_ref, list(i.outputs())[0]]
            self.key_l[name] = loss
            self.key_c[name] = classification

    def FinalizeOutput(self):
        self.Notify(">--------- Final Output Mapping ---------<")
        tmp = {}
        for i in self.key_o:
            key = i.lstrip("O_")
            if isinstance(self.key_c[key], bytes):
                self.key_c[key] = bool(self.key_c[key].decode().split("->")[0])
                self.key_l[key] = str(self.key_l[key].decode()).split("->")[0]
                self.GetNodeAsOutput(key, self.key_o[i][0], self.key_l[key], self.key_c[key]) 

            out = self.key_o[i]
            if out == -1:
                continue

            self.Notify("NODE REFERENCE: " + out[0] + " FEATURE NAME: " + key + " LOSS: " + self.key_l[key] + " CLASSIFICATION: " + str(self.key_c[key]))
            setattr(self, i, None)
            setattr(self, "L_" + key, self.key_l[key])
            setattr(self, "C_" + key, self.key_c[key])
            self._model.graph.registerOutput(out[1])
            tmp[i] = self._it
            self._it += 1
        self._it = None
        self.key_o = tmp
        self._model.graph.makeMultiOutputIntoTuple()

    def train(self):
        self._model.train()

    def eval(self):
        self._model.eval()
    
    def __call__(self, **kargs):
        out = self._model(**kargs)
        for i in self.key_o:
            setattr(self, i, out[self.key_o[i]])

    def to(self, device):
        self._model.to(device)




class ModelComparison(Reconstructor, Directories, WriteDirectory):

    def __init__(self):
        self.VerboseLevel = 0
        self.Verbose = True 
        self.Caller = "ModelComparison"
        self.Threads = 12
        self.Device = "cuda"
        self.Models = {}
        self.ModelDirectoryONNX = {}
        self.ModelDirectoryTScript = {}
        self.OutputCollection = {}
        self.OutputCollection["E_Truth"] = {}
        self.OutputCollection["N_Truth"] = {}
        self._init = None

    def AddModel(self, model, CustomMap = []):
        name =  model.ModelDirectoryName
        self.Models[name] = model
        self.ModelDirectoryONNX[name] = self.ListFilesInDir(model.SourceDirectory + "/" + name + "/ONNX")
        self.ModelDirectoryTScript[name] = self.ListFilesInDir(model.SourceDirectory + "/" + name + "/TorchScript")
        self.ModelDirectoryTScript["_MAP_"+name] = CustomMap

    def _ImportModelsTS(self):
        for name in self.ModelDirectoryTScript:
            if name.startswith("_MAP_"):
                continue 
            files = self.ModelDirectoryTScript[name]
            maps = self.ModelDirectoryTScript["_MAP_" + name]
            self.OutputCollection[name] = {}
            
            Epochs = []
            self.Notify("IMPORTING MODEL TORCH SCRIPT:" + name)
            for i in files:
                epoch = i.split("/")[-1].split("_")
                f = epoch[-1].replace(".pt", "")
                self.Notify("-> Imported EPOCH " + str(epoch[1]) + " / " + str(f))
                M = TorchScriptModel(i, Device = self.Device, maps = maps) 
                M.FinalizeOutput()
                Epochs += [[epoch[1], M]]
                self.OutputCollection[name][epoch[1]] = {}
                break
            self.ModelDirectoryTScript[name] = Epochs

    def _Loop(self, Sample, feat, pt, eta, phi, e, Edge):
        def Clean(inpt, out = []):
            for i in inpt:
                if len(i.shape) != 0:
                    out += Clean(i)
                else:
                    out.append(float(i))
            return out



        out = []
        for i in list(Sample.values()):
            i.to(device = self.Device)
            self.Sample = i
            if Edge:
                val = self.MassFromEdgeFeature(feat, "N_" + pt, "N_" + eta, "N_" + phi, "N_" + e)
            else:
                val = self.MassFromNodeFeature(feat, "N_" + pt, "N_" + eta, "N_" + phi, "N_" + e)
            
            out += Clean(val)
        return out

    def _Rebuild(self, Sample, pt, eta, phi, e, varname, Level):
        if self._init == None: 
            self._ImportModelsTS()
        
        if Level == "Edge":
            key = "E_Truth"
            edge = True
        else:
            key = "N_Truth"
            edge = False

        for i in self.ModelDirectoryTScript:
            if i.startswith("_MAP_"):
                continue 
         
            if varname not in self.OutputCollection[key]:
                self.TruthMode = True
                self.OutputCollection[key][varname] = self._Loop(Sample, varname, pt, eta, phi, e, edge) 
                self.TruthMode = False 
           
                H = TH1F()
                H.xData = self.OutputCollection[key][varname]
                H.Title = varname
                H.xTitle = "Mass (GeV)"
                H.xBins = 100
                H.xMax = 300
                H.xMin = 100
                H.Filename = "TRUTH_" + varname
                H.SaveFigure("./")

            for smpl in list(Sample.values()):
                break
            
            smpl.to(device = self.Device)
            self.Sample = smpl
            
            for ep in range(len(self.OutputCollection[i])):
                epoch = self.ModelDirectoryTScript[i][ep][0]
                self.Model = self.ModelDirectoryTScript[i][ep][1]
                self._init = False
                self.InitializeModel()
                self.OutputCollection[i][epoch][varname] = self._Loop(Sample, varname, pt, eta, phi, e, edge)

                H = TH1F()
                H.xData = out
                H.Title = varname
                H.xTitle = "Mass (GeV)"
                H.xBins = 100
                H.xMax = 300
                H.xMin = 100
                H.Filename = varname + "_" + str(epoch)
                H.SaveFigure("./")


    def RebuildMassEdge(self, Sample, varname_pt, varname_eta, varname_phi, varname_energy, ModelOutputVaribleName):
        self._Rebuild(Sample, varname_pt, varname_eta, varname_phi, varname_energy, ModelOutputVaribleName, "Edge")

    def RebuildMassNode(self, Sample, varname_pt, varname_eta, varname_phi, varname_energy, ModelOutputVaribleName):
        self._Rebuild(Sample, varname_pt, varname_eta, varname_phi, varname_energy, ModelOutputVaribleName, "Node")
