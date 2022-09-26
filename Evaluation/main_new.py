from RebuildingMetrics import ModelEvaluator

x = ModelEvaluator()
x.VerboseLevel = 2
x.AddFileTraces("./BasicBaseLineChildren")
x.Compile()
