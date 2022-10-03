import os
from glob import glob
from sklearn.metrics import roc_curve, auc
import numpy as np
import torch

class Tools:
    def UnNestList(self, inpt):
        if isinstance(inpt, list) == False:
            return [inpt]
        out = []
        for i in inpt:
            out += self.UnNestList(i)
        return out

    def UnNestDict(self, inpt):
        if isinstance(inpt, dict) == False:
            return inpt        
        out = []
        for i in inpt:
            out += self.UnNestDict(inpt[i])
        return out 
    
    def CollectKeyNestDict(self, inpt, search):
        if isinstance(inpt, dict) == False:
            return []
        out = []
        for i in inpt:
            if i == search:
                return inpt[i] if isinstance(inpt[i], list) else [inpt[i]]
            out += self.CollectKeyNestDict(inpt[i], search)
        return out

    def pwd(self):
        return os.getcwd()

    def abs(self, directory):
        return os.path.abspath(directory)

    def mkdir(self, directory):
        try:
            os.makedirs(directory)
        except FileExistsError:
            pass

    def ListFilesInDir(self, directory):
        return [i.split("/")[-1] for i in glob(directory + "/*")]

    def TraverseDictionary(self, inpt, path):
        if isinstance(inpt, dict) == False:
            return inpt
        split = path.split("/")
        if len(split) == 1:
            return inpt[split[0]]
        return self.TraverseDictionary(inpt[split[0]], "/".join(split[1:]))


class Metrics:

    def MakeROC(self, feature):
        truth = self.ROC[feature]["truth"]
        truth = torch.cat(truth, dim = 0).view(-1)
        truth = truth.detach().cpu().numpy()
        
        p_score = self.ROC[feature]["pred_score"]
        p_score = torch.cat([p.softmax(dim = 1).max(1)[0] for p in p_score], dim = 0)
        p_score = p_score.detach().cpu().numpy()
        
        fpr, tpr, _ = roc_curve(truth, p_score)
        auc_ = auc(fpr, tpr)

        self.ROC[feature]["fpr"] += fpr.tolist()
        self.ROC[feature]["tpr"] += tpr.tolist()
        self.ROC[feature]["auc"].append(float(auc_))

        return self.ROC
   

    def ClosestParticle(self, tru, pred):

        res = []
        if len(tru) == 0:
            return res
        p = pred.pop(0)
        max_tru, min_tru = max(tru), min(tru)
        col = True if p <= max_tru and p >= min_tru else False

        if col == False:
            if len(pred) == 0:
                return res
            return self.ClosestParticle(tru, pred)

        diff = [ abs(p - t) for t in tru ]
        tru.pop(diff.index(min(diff)))
        res += self.ClosestParticle(tru, pred)
        res.append(p)
        return res 
    
    def ParticleEfficiency(self, pred, truth, proc):
        t_, p_ = [], []
        t_ += truth
        p_ += pred 

        p = self.ClosestParticle(t_, p_)
        p_l, t_l = len(p), len(truth)

        perf = float(p_l/t_l)*100

        out = {"Prc" : proc, "%" : perf, "nrec" : p_l, "ntru" : t_l}
        
        return out
