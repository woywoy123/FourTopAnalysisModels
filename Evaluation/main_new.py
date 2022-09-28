from RebuildingMetrics import ModelEvaluator

OutputMap = [{"name" : "edge", "node" : "366"}]


x = ModelEvaluator()
x.MakeTrainingPlots = False
x.MakeSampleNodesPlots = False
x.MakeStaticHistogramPlot = False
x.RebuildSize = 100
x.RebuildRandom = True
x.VerboseLevel = 3
x.CompareToTruth = True
x.Device = "cuda"

Dir = "./"
x.AddFileTraces(Dir + "BasicBaseLineChildren")
x.AddModel(Dir + "BasicBaseLineChildren/Models/BaseLine_MRK1/")
#x.AddModel(Dir + "BasicBaseLineChildren/Models/BaseLine_MRK2/")
#x.AddModel(Dir + "BasicBaseLineChildren/Models/BaseLine_MRK3/")
#x.AddModel(Dir + "BasicBaseLineChildren/Models/BaseLine_MRK4/")
#x.AddModel(Dir + "BasicBaseLineChildren/Models/BaseLine_MRK5/")
#x.AddModel(Dir + "BasicBaseLineChildren/Models/BaseLine_MRK6/")
#x.AddModel(Dir + "BasicBaseLineChildren/Models/BaseLine_MRK7/")
#x.AddTorchScriptModel("BaseLine_MRK1", OutputMap)
x.ROCCurveFeature("edge")
x.Compile()
