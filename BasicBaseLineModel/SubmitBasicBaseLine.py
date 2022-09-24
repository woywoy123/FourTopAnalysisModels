from AnalysisTopGNN.Generators import Analysis 
from AnalysisTopGNN.Submission import Condor
from AnalysisTopGNN.Events import Event, EventGraphTruthJetLepton, EventGraphTruthTopChildren
from EventFeatureTemplate import ApplyFeatures
from V3.BasicBaseLine import BasicBaseLineRecursion

Submission = Condor()
Submission.ProjectName = "BasicBaseLineChildren"
GeneralDir = "/CERN/CustomAnalysisTopOutput/"

#GeneralDir = "/nfs/dust/atlas/user/<...>/SamplesGNN/SmallSample/"
#GeneralDir = "/nfs/dust/atlas/user/<...>/SamplesGNN/CustomAnalysisTopOutput/"

def EventLoaderConfig(Name, Dir):
    Ana = Analysis()
    Ana.Event = Event
    Ana.Threads = 12
    Ana.EventCache = True 
    Ana.Tree = "nominal"
    Ana.InputSample(Name, GeneralDir + Dir)
    Submission.AddJob(Name, Ana, "64GB", "24h")

def DataLoaderConfig(Name):
    Ana = Analysis()
    Ana.EventGraph = EventGraphTruthTopChildren
    Ana.DataCache = True
    Ana.FullyConnect = True
    Ana.SelfLoop = True
    Ana.DumpHDF5 = True
    Ana.Threads = 12
    Ana.InputSample(Name)
    ApplyFeatures(Ana, "TruthChildren")
    Ana.DataCacheOnlyCompile = [Name]
    Submission.AddJob("Data_" + Name, Ana, "64GB", "24h", [Name])

def ModelConfig(Name):
    TM = Analysis()
    TM.RunName = Name
    TM.ONNX_Export = True
    TM.TorchScript_Export = True
    TM.kFold = 10
    TM.Threads = 4
    TM.Device = "cuda"
    TM.Epochs = 100
    TM.BatchSize = 20
    TM.Model = BasicBaseLineRecursion()
    return TM


Submission.SkipEventCache = False
Submission.SkipDataCache = False

# ====== Event Loader ======== #
EventLoaderConfig("ttbar", "ttbar")
EventLoaderConfig("SingleTop", "t")
EventLoaderConfig("BSM4Top", "tttt")
#EventLoaderConfig("Zmumu", "Zmumu")


# ====== Data Loader ======== #
DataLoaderConfig("ttbar")
DataLoaderConfig("SingleTop")
DataLoaderConfig("BSM4Top")
#DataLoaderConfig("Zmumu")

# ====== Merge ======= #
Smpl = ["Data_BSM4Top", "Data_ttbar", "Data_SingleTop"] #, "Data_Zmumu"]
Loader = Analysis()
Loader.InputSample("ttbar")
Loader.InputSample("SingleTop")
Loader.InputSample("BSM4Top")
#Loader.InputSample("Zmumu")
Loader.MergeSamples = True
Loader.GenerateTrainingSample = True
Loader.ValidationSize = 90
Submission.AddJob("Sample", Loader, "64GB", "96h", Smpl)

# ======= Model to Train ======== #
TM1 = ModelConfig("BaseLine_MRK1")
TM1.LearningRate = 0.01
TM1.WeightDecay = 0.01
TM1.SchedulerParams = {"gamma" : 0.5}
TM1.DefaultScheduler = "ExponentialR"
Submission.AddJob("BasicBaseLine_MRK1", TM1, "12GB", "48h", ["Sample"])

TM2 = ModelConfig("BaseLine_MRK2")
TM2.LearningRate = 0.001
TM2.WeightDecay = 0.01
TM2.SchedulerParams = {"gamma" : 1.0}
TM2.DefaultScheduler = "ExponentialR"
Submission.AddJob("BasicBaseLine_MRK2", TM2, "12GB", "48h", ["Sample"])

TM3 = ModelConfig("BaseLine_MRK3")
TM3.LearningRate = 0.001
TM3.WeightDecay = 0.001
TM3.SchedulerParams = {"gamma" : 1.5}
TM3.DefaultScheduler = "ExponentialR"
Submission.AddJob("BasicBaseLine_MRK3", TM3, "12GB", "48h", ["Sample"])

TM4 = ModelConfig("BaseLine_MRK4")
TM4.LearningRate = 0.01
TM4.WeightDecay = 0.001
TM4.SchedulerParams = {"gamma" : 2.0}
TM4.DefaultScheduler = "ExponentialR"
Submission.AddJob("BasicBaseLine_MRK4", TM4, "12GB", "48h", ["Sample"])

TM5 = ModelConfig("BaseLine_MRK5")
TM5.LearningRate = 0.001
TM5.WeightDecay = 0.001
TM5.SchedulerParams = {"base_lr" : 0.000001, "max_lr" : 0.1}
TM5.DefaultScheduler = "CyclicLR"
Submission.AddJob("BasicBaseLine_MRK5", TM5, "12GB", "48h", ["Sample"])

TM6 = ModelConfig("BaseLine_MRK6")
TM6.LearningRate = 0.001
TM6.WeightDecay = 0.001
TM6.DefaultScheduler = None
Submission.AddJob("BasicBaseLine_MRK6", TM6, "12GB", "48h", ["Sample"])

TM7 = ModelConfig("BaseLine_MRK7")
TM7.LearningRate = 0.001
TM7.WeightDecay = 0.001
TM7.DefaultOptimizer = "SGD"
TM7.DefaultScheduler = None
Submission.AddJob("BasicBaseLine_MRK7", TM7, "12GB", "48h", ["Sample"])

#Submission.LocalDryRun()
Submission.DumpCondorJobs()
