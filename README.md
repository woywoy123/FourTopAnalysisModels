# FourTopAnalysisModels

## Introduction
This repository is going to be used in conjuction with the ``FourTopsAnalysis`` which provides the framework to evaluate the models. Here models are tested and debugged until they reach maturity. Furthermore, the repository will also serve as a place to track samples being tested and used for the evaluation process. 

## Current Samples Being Tested:
### At Truth Jet Level
- Zmumu
- ttbar
- BSM4Tops: 1500 GeV Injection
- SingleTop

## Models Implemented
This section is dedicated towards the different model implementations and provides a short description of their underlying mechanics.

### BasicBaseLineModelTruthJet
#### Description/Algorithm:
In this model first the individual node masses are calculated from their cartesian coordinates (relevant for summing four vectors). Subsequently, a MLP (```_Node```) encodes the node's properties, which are then parsed into a message passing algorithm. During message passing, three MLPs are used, ```_mass```, ```_Edge``` and ```_isedge```, where the initial MLP encodes the mass of the edge, followed by a delta R and leptonic encoder, and finally an edge classifier taking in the prior two MLP encoders as input.

During message aggregation, the edge classifier (``_isedge``) blocks non-conformal edges by setting the incoming message value to zero, allowing for four vector aggregation to be performed, such that ```_mass``` can learn higher resonances. The information of both ```_mass``` and ```_Node``` are then exploited to predict whether the given node and its neighbors are members of a common top-quark.

An additional MLP (```_node_m```) uses the aggregated incoming edge information to classify whether nodes are members of the Z' resonance and computes the invariant mass of particles identified as Z' nodes. A final MLP (```_signal```) classifies the entire graph (with appropriate inputs) as either being signal (Z') like. 

#### Node Inputs:
- eta, energy, pt, phi, mass, islep (isLepton), charge
#### Graph Inputs:
- mu, met (missing-et), phi, pileup, njets, nlep

#### Edge Output/Loss:
- ```O_edge```: Predicts the topology of the graph
- ```L_edge```: CEL (Cross Loss Entropy)

#### Node Outputs/Loss:
- ```O_from_res```: Predicts whether the given particle node is a member of the injected Z' resonance
- ```L_from_res```: CEL (Cross Loss Entropy)

- ```O_from_top```: Predicts if particle node originates from a top-quark
- ```L_from_top```: CEL (Cross Loss Entropy)

#### Graph Outputs/Loss:
- ```O_signal_sample```: Predicts if the event is contains a resonance, regardless if the nodes are not identified
- ```L_signal_sample```: CEL (Cross Loss Entropy)

#### Multi-Layer-Perceptrons (MLP):
##### No activation functions - Only Linear Transformation 
- ```_Node```: 7 - 256 - 1024 - 2048 
- ```_Edge```: 2 - 256 - 1024 - 2048 
- ```_mass```: 1 - 1024 - 1024 - 2048 
- ```_node_m```: 4x2048 - 1024 - 2048 
- ```_mass```: 8 - 256 - 256 - 256 - 2 

##### Non adjustable bias layers. 
- ```_isedge```: 2x2048 -> 0.5x2048 -> ReLU -> 0.5x2048 -> 2
- ```_istop```:  2x2048 -> 0.5x2048 -> ReLU -> 0.5x2048 -> 2
- ```_ResSw```:  2048 -> 0.5x2048 -> ReLU -> 0.5x2048 -> 2
- ```_fromRes```: 3x2048 -> 0.5x2048 -> ReLU -> 0.5x2048 -> 2

## Current Training Schedule
- LR: Learning Rate 
- WD: Weight Decay
- BS: Batch Size
- S: Scheduler 

### Model: BasicBaseLineModelTruthJet
#### Common Parameters
- Fraction of sample used for training: 20%
- kFold: 10
- Epochs: 100
- BS: 50
- Minimizer: Adam (except for TM7)

#### Specific Parameters:
- TM1: LR = 0.01,  WD = 0.01,  S = ExponentialR (gamma 0.9)
- TM2: LR = 0.001, WD = 0.01,  S = ExponentialR (gamma 0.9)
- TM3: LR = 0.001, WD = 0.001, S = ExponentialR (gamma 0.9)
- TM4: LR = 0.01,  WD = 0.001, S = ExponentialR (gamma 0.9)

- TM5: LR = 0.001, WD = 0.001, S = CyclicLR ("base_lr" : 0.000001, "max_lr" : 0.1)
- TM6: LR = 0.001, WD = 0.001, S = None, Static LR
- TM7: LR = 0.001, WD = 0.001, S = None, Static LR, Minimizer: SGD
