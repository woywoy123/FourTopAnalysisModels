from TrainingPlots import Training
from AllPlots import All, Train, Test
   
class FigureContainer:

    def __init__(self):
        self.OutputDirectory = None
        self.training = Training()
        self.all = All()
        self.test = Test()
        self.train = Train()

    def AddEpoch(self, epoch, vals):
        if "training" in vals:
            self.training.AddEpoch(epoch, vals["training"])
        elif "all" in vals:
            self.all.AddEpoch(epoch, vals["all"])
        elif "train" in vals:
            self.train.AddEpoch(epoch, vals["train"])
        elif "test" in vals:
            self.test.AddEpoch(epoch, vals["test"])

    def Compile(self):
        self.training.Compile(self.OutputDirectory)
        self.all.Compile(self.OutputDirectory, "all")
        self.test.Compile(self.OutputDirectory, "test")
        self.train.Compile(self.OutputDirectory, "train")




