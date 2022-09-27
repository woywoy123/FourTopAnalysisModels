from RebuildingMetrics import ModelEvaluator

OutputMap = [{"name" : "edge", "node" : "isEdge"}]


x = ModelEvaluator()
x.MakeTrainingPlots = False
x.MakeSampleNodesPlots = False
x.RebuildSize = 100
x.RebuildRandom = True
x.VerboseLevel = 2
x.Device = "cuda"
x.AddFileTraces("./BasicBaseLineChildren")
x.AddModel("./BasicBaseLineChildren/Models/BaseLine_MRK1/")
x.DefineModelOutputs("BaseLine_MRK1", OutputMap)
x.Compile()
