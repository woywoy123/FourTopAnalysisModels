from AnalysisTopGNN.Generators import Analysis 
from AnalysisTopGNN.Submission import Condor
from AnalysisTopGNN.Events import Event, EventGraphTruthJetLepton
from EventFeatureTemplate import ApplyFeatures
from BasicBaseLine import BasicBaseLineTruthJet

GeneralDir = "/CERN/CustomAnalysisTopOutputTest/"
TopBackgrounds = [GeneralDir + "t", GeneralDir + "ttbar"]
Signal = [GeneralDir + "tttt"]

ttbar = Analysis()
ttbar.EventImplementation = Event
ttbar.CompileSingleThread = False
ttbar.CPUThreads = 4
ttbar.EventCache = True
ttbar.InputSample("ttbar", TopBackgrounds[1])

SingleTop = Analysis()
SingleTop.EventImplementation = Event
SingleTop.CompileSingleThread = False
SingleTop.CPUThreads = 4
SingleTop.EventCache = True
SingleTop.InputSample("SingleTop", TopBackgrounds[0])

bsm4top = Analysis()
bsm4top.EventImplementation = Event
bsm4top.CompileSingleThread = False
bsm4top.CPUThreads = 4
bsm4top.EventCache = True
bsm4top.InputSample("BSM4Top", Signal[0])


ttbarData = Analysis()
ttbarData.ProjectName = "BaseLineTopAnalysis"
ttbarData.EventGraph = EventGraphTruthJetLepton
ttbarData.DataCache = True
ApplyFeatures(ttbarData)
ttbarData.DataCacheOnlyCompile = ["ttbar"]

SingleTopData = Analysis()
SingleTopData.EventGraph = EventGraphTruthJetLepton
SingleTopData.DataCache = True
ApplyFeatures(SingleTopData)
SingleTopData.DataCacheOnlyCompile = ["SingleTop"]

bsm4topData = Analysis()
bsm4topData.EventGraph = EventGraphTruthJetLepton
bsm4topData.DataCache = True
ApplyFeatures(bsm4topData)
bsm4topData.DataCacheOnlyCompile = ["BSM4Top"]

Loader = Analysis()
Loader.Device = "cuda"
Loader.GenerateTrainingSample = True
Loader.RebuildTrainingSample = True 
Loader.ProjectName = "BaseLineTopAnalysis"

TrainModel = Analysis()
TrainModel.RunName = "BaseLineTruthJet"
TrainModel.ONNX_Export = True
TrainModel.TorchScript_Export = True
TrainModel.kFold = 10
TrainModel.Device = "cuda"
TrainModel.Epochs = 10
TrainModel.Debug = False
TrainModel.Model = BasicBaseLineTruthJet()


Submission = Condor()
Submission.ProjectName = "BaseLineTopAnalysis"
Submission.DisableRebuildTrainingSample = False
Submission.DisableDataCache = False 
Submission.DisableEventCache = False
Submission.AddJob("ttbar", ttbar, "1GB", "1h")
Submission.AddJob("SingleTop", SingleTop, "1GB", "1h")
Submission.AddJob("Signal", bsm4top, "1GB", "1h")
Submission.AddJob("ttbar_data", ttbarData, "1GB", "1h", ["ttbar"])
Submission.AddJob("SingleTop_data", SingleTopData, "1GB", "1h", ["SingleTop"])
Submission.AddJob("base4top_data", bsm4topData, "1GB", "1h", ["Signal"])
Submission.AddJob("Sample", Loader, "1GB", "1h", ["SingleTop_data", "base4top_data", "ttbar_data"])
Submission.AddJob("TrainingBaseLine", TrainModel, "2GB", "1h", ["Sample"])
#Submission.LocalDryRun()
Submission.DumpCondorJobs()