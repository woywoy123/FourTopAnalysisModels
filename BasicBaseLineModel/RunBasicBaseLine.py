from AnalysisTopGNN.Tools.ModelTesting import CreateWorkspace, KillCondition, OptimizerTemplate
from AnalysisTopGNN.Generators import Analysis
from AnalysisTopGNN.Events import Event, EventGraphTruthJetLepton
from BasicBaseLine import *
from EventFeatureTemplate import TruthJets

def BaseLineModelTruthJet(Files, Names, CreateCache):
    CreateCache = False
    Features = TruthJets()
    DL = CreateWorkspace(Files, Features, CreateCache, 100, Names, "TruthJetLepton", True)
    samples = DL.TrainingSample
    k = 14 
    #su = 0
    #for i in samples:
    #    su += len(samples[i])
    #    print(i, len(samples[i]))
    #print(su)
    #exit()

    Model = BasicBaseLineTruthJet()
    Op = OptimizerTemplate(DL, Model)
    Op.LearningRate = 0.0001
    Op.WeightDecay = 0.001
    Op.DefineOptimizer()

    kill = {}
    kill |= {"edge" : "R"}
    #kill |= {"from_res" : "C"}
    #kill |= {"signal_sample": "C"}
    #kill |= {"from_top": "C"}
    KillCondition(kill, 50, Op, samples[k], 100000, sleep = 2, batched = 3)


def BaseLineModelTruthJetAnalysis():
    GeneralDir = "/CERN/CustomAnalysisTopOutputTest/"
    TopBackgrounds = [GeneralDir + "t", GeneralDir + "ttbar"]
    Signal = [GeneralDir + "tttt"]
    
    A = Analysis()
    
    # Config Settings
    A.ProjectName = "TopEvaluation"
    A.EventImplementation = Event
    A.CompileSingleThread = False
    A.CPUThreads = 4
    A.EventCache = False
    A.DataCache = False
    A.GenerateTrainingSample = False
    A.TrainWithoutCache = True

    A.EventGraph = EventGraphTruthJetLepton
    A.TrainingSampleSize = 80
    A.Device = "cuda"
    A.SelfLoop = True
    A.FullyConnect = True
    #A.NEvent_Stop = 100

    A.Model = BasicBaseLineTruthJet()
    A.LearningRate = 0.0001
    A.WeightDecay = 0.0001
    A.kFold = 2
    A.Epochs = 3
    A.BatchSize = 20
    A.RunName = "BasicBaseLineTruthJet"
    A.ONNX_Export = True
    A.TorchScript_Export = True

    Features = TruthJets()
    for i in Features:
        base = "_".join(i.split("_")[1:])
        fx = Features[i]
        
        if "EF" in i:
            A.AddEdgeFeature(base, fx)
        elif "NF" in i:
            A.AddNodeFeature(base, fx)
        elif "GF" in i:
            A.AddGraphFeature(base, fx)

        elif "ET" in i:
            A.AddEdgeTruth(base, fx)
        elif "NT" in i:
            A.AddNodeTruth(base, fx)
        elif "GT" in i:
            A.AddGraphTruth(base, fx)


    A.InputSample("TopBackground", TopBackgrounds)
    A.InputSample("SignalSample", Signal)
    A.Launch()

if __name__ == "__main__":
    GeneralDir = "/CERN/CustomAnalysisTopOutputTest/"
    Files = [GeneralDir + "tttt/QU_0.root", GeneralDir + "t/QU_0.root"]
    Names = ["tttt", "t"]
    CreateCache = True
    BaseLineModelTruthJet(Files, Names, CreateCache)
    #BaseLineModelTruthJetAnalysis()
