#from RebuildingMetrics import ModelEvaluator
#
#OutputMap = [{"name" : "edge", "node" : "366"}]
#
#
#x = ModelEvaluator()
#x.MakeTrainingPlots = True
#x.MakeTestPlots = True
#x.MakeSampleNodesPlots = True
#x.MakeStaticHistogramPlot = False
#x.RebuildSize = 1
#x.chnk = 1000
#x.RebuildRandom = True
#x.VerboseLevel = 3
#x.CompareToTruth = True
#x.Device = "cuda"
#
#Dir = "/CERN/AnalysisModelTraining/PreviousVersions/TruthTopChildrenModels/"
#x.AddFileTraces(Dir + "BasicBaseLineChildren")
#x.AddModel(Dir + "BasicBaseLineChildren/Models/BasicBaseLineNominal_MRK1/")
##x.AddModel(Dir + "BasicBaseLineChildren/Models/BaseLine_MRK2/")
##x.AddModel(Dir + "BasicBaseLineChildren/Models/BaseLine_MRK3/")
##x.AddModel(Dir + "BasicBaseLineChildren/Models/BaseLine_MRK4/")
##x.AddModel(Dir + "BasicBaseLineChildren/Models/BaseLine_MRK5/")
##x.AddModel(Dir + "BasicBaseLineChildren/Models/BaseLine_MRK6/")
##x.AddModel(Dir + "BasicBaseLineChildren/Models/BaseLine_MRK7/")
#
#
#x.AddModel(Dir + "BasicBaseLineChildren/Models/BasicBaseLineRecursion_MRK1/")
##x.AddModel(Dir + "BasicBaseLineChildren/Models/BaseLine_MRK2/")
##x.AddModel(Dir + "BasicBaseLineChildren/Models/BaseLine_MRK3/")
##x.AddModel(Dir + "BasicBaseLineChildren/Models/BaseLine_MRK4/")
##x.AddModel(Dir + "BasicBaseLineChildren/Models/BaseLine_MRK5/")
##x.AddModel(Dir + "BasicBaseLineChildren/Models/BaseLine_MRK6/")
##x.AddModel(Dir + "BasicBaseLineChildren/Models/BaseLine_MRK7/")
#
#
#
##x.AddTorchScriptModel("BaseLine_MRK1", OutputMap)
#x.ROCCurveFeature("edge")
#x.MassFromEdgeFeature("edge")
#
#
#
#
#x.Compile()

Dir = "/CERN/AnalysisModelTraining/PreviousVersions/TruthTopChildrenModels/"
OutputMap = [{"name" : "edge", "node" : "366"}]
from Metrics import ModelEvaluator


x = ModelEvaluator()
x.BuildDataPercentage = 1
x.MakeTrainingPlots = False
x.AddFileTraces(Dir + "BasicBaseLineChildren")
x.AddModel(Dir + "BasicBaseLineChildren/Models/BasicBaseLineRecursion_MRK1")
x.AddModel(Dir + "BasicBaseLineChildren/Models/BasicBaseLineNominal_MRK1")
x.DefineTorchScriptModel("BasicBaseLineRecursion_MRK1", OutputMap)
x.ROCCurveFeature("edge")
x.MassFromEdgeFeature("edge")
x.MassFromNodeFeature("...")
x.Compile("./ModelEvaluator")
