import torch
from AnalysisTopGNN.Plotting import TH1F, CombineTH1F
from torch.nn import Sequential as Seq, Linear, ReLU, Sigmoid, Tanh
from torch_geometric.nn import MessagePassing
import random
    
class MLP(MessagePassing):

    def __init__(self):
        super().__init__()
        end = 1024
        self.MLP = Seq(Linear(1, end), Sigmoid(), Linear(end, end), Linear(end, end), Sigmoid(), Linear(end, 2))
        
    def forward(self, inpt):
        return self.MLP(inpt)

def Train(model, inpt, truth, learnRate = 0.001, epoch = 100):

    optimizer = torch.optim.Adam(model.parameters(), lr = learnRate)
    model.train()
    model = model.to("cuda")
    loss = torch.nn.CrossEntropyLoss()
    
    loss_dic = {}
    for ep in range(epoch):
        loss_dic[ep] = []
        for i, t in zip(inpt, truth):
            optimizer.zero_grad()
            out = model(i.view(1, -1)[0])
            
            out = out.view(1, -1)
            ls = loss(out, t.view(-1)) 
            
            ls.backward() 
            optimizer.step()
            
            loss_dic[ep].append(ls.item())

def Gaussian(mean = 0, stdev = 1, classifier = 0):

    x = torch.normal(mean = mean, std = stdev, size = (1, 10000)).tolist()[0]
    truth = [classifier]*len(x)
    
    t = TH1F()
    t.Title = "mean; " + str(mean) + " stdev; " + str(stdev)
    t.xBins = 100
    t.xMin = -5
    t.xMax = 5
    t.xData = x
    t.Filename = "Gaussian"+ str(mean) + "_" + str(stdev)
    t.SaveFigure("./Gaussians/")
    
    return t, x, truth


def CombinedTH1(t1, t2, distance):

    H = CombineTH1F(Title = "Superimposed Gaussian Distributions")
    t2.Color = "red"
    H.Histograms = [t1, t2]
    H.Filename = "Gaussians" + str(distance)
    H.xBins = 100
    H.SaveFigure("./Distance/" + str(distance) + "/Gaussian")



if __name__ == '__main__':

    
    hist1, x1, t1 = Gaussian(0, 1, 0)
    hist2, x2, t2 = Gaussian(1, 1, 1)
    
    CombinedTH1(hist1, hist2, 1)
    data = []
    data += x1
    data += x2 

    truth = []
    truth += t1
    truth += t2

    c = list(zip(data, truth))
    random.shuffle(c)
    data, truth = zip(*c)
    data = torch.tensor(data, device = "cuda")
    truth = torch.tensor(truth, device = "cuda")
    
    model = MLP()
    
    Train(model, data, truth)
    



