import argparse
import os
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pickle
import math
from Bio.SubsMat import MatrixInfo as matrices
import time
import copy

timer = time.time()
batchSize = 1000
hiddenLayers = 200
weightNonbinding = 0.4
weightBinding = 0.6
learning_rate = 3e-3
epochs = 200
device = torch.device('cpu')
crossValidation = False
predCutoff = 0.4
momentum=0.9
#device = torch.device('cuda')
blosumScalar = 1
modelPath = "initialModel"
trainedModelPath = 'trainedModel.pickle'

parser = argparse.ArgumentParser(description='Load and analyse protein binding site data')
parser.add_argument('--fastaFolder', help = "Path to folder containing fasta files", type = str)
parser.add_argument('--snapFolder', help = "Path to folder containing snap files", type = str)
parser.add_argument('--bindingResidues', help = "Path to binding residues files", type = str)
parser.add_argument('--pickleTrain', help = "Path to pickle file with train data structures", type=str)
parser.add_argument('--pickleLabel', help = "Path to pickle file with label data structures", type=str)
parser.add_argument('--trainedModel', help='Path to pre-trained model', type=str)

args = parser.parse_args()

fastaFolder = args.fastaFolder
snapFolder = args.snapFolder
bindingResiduesFile = args.bindingResidues
pickleFileTrain = args.pickleTrain
pickleFileLabel = args.pickleLabel
trainedmodel = args.trainedModel
if trainedmodel != None:
    testmode = True
else:
    testmode = False

blosum62 = copy.deepcopy(matrices.blosum62)
blosum62scaled = copy.deepcopy(matrices.blosum62)
blosumCutoffsDict = {
  -4 : 90,
  -3 : 80,
  -2 : 70,
  -1 : 60,
  0 : 50,
  1 : 40,
  2 : 30,
  3 : 0
}

blosumScalingDict = {
  -4 : -100,
  -3 : -75,
  -2 : -50,
  -1 : -25,
  0 : 0,
  1 : 33,
  2 : 66,
  3 : 100
}

for key in blosum62scaled:
  if blosum62scaled[key] in blosumScalingDict:
    blosum62scaled[key] = blosumScalingDict[blosum62scaled[key]]

def loadFastaFiles(fastaFolder):
  proteinSeqDict = {}
  for filename in os.listdir(fastaFolder):
    with open(os.path.join(fastaFolder, filename), 'r') as fastaFile:
      header = fastaFile.readline()
      header = header.strip('\n')
      sequence = fastaFile.readline()
      proteinSeqDict[header[1:]] = sequence
  return proteinSeqDict

def loadSnapCalcFeature(snapFolder, blosum62, blosum62scaled, blosumCutoffsDict):
  aaOrder = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
  snapScoreDict = {}
  snapConfDict = {}
  featureDict = {}
  feature2Dict = {}
  for filename in os.listdir(snapFolder):
    aaIndex = 0
    proteinID = os.path.basename(filename)[:-6]
    snapScoreDict[proteinID] = []
    snapConfDict[proteinID] = []
    featureDict[proteinID] = []
    feature2Dict[proteinID] = []
    posAAscores = []
    posAAconf = []
    features = []
    features2 = []
    with open(os.path.join(snapFolder, filename), 'r') as snapFile:
      for line in snapFile:
        splitLine = line.split()
        score = splitLine[len(splitLine)-1]
        posAAinfo = splitLine[0]
        posAAid = posAAinfo[0]
        mutAAid = posAAinfo[-1]
        posAAindex = aaOrder.index(posAAid)
        mutAAindex = aaOrder.index(mutAAid)
        if posAAindex == aaIndex:
          aaIndex += 1
          posAAscores.append(0)
          posAAconf.append(0)
          features.append(0)
          features2.append(0)
          features2.append(0)
        if score[-1] == '%':
          score = score[:-1]
        score = int(score)
        if len(posAAscores) > len(posAAconf):
          posAAconf.append(score)
          if aaIndex < 19:
            aaIndex += 1
            if(aaIndex == posAAindex == 19):
                posAAscores.append(0)
                posAAconf.append(0)
                features.append(0)
                features2.append(0)
                features2.append(0)
                snapScoreDict[proteinID].append(posAAscores)
                snapConfDict[proteinID].append(posAAconf)
                featureDict[proteinID].append(features)
                feature2Dict[proteinID].append(features2)
                posAAscores = []
                posAAconf = []
                features = []
                features2 = []
                aaIndex = 0
          else:
            snapScoreDict[proteinID].append(posAAscores)
            snapConfDict[proteinID].append(posAAconf)
            featureDict[proteinID].append(features)
            feature2Dict[proteinID].append(features2)
            posAAscores = []
            posAAconf = []
            features = []
            features2 = []
            aaIndex = 0
        else:
          posAAscores.append(score)
          if posAAindex < mutAAindex:
            firstAA = mutAAid
            secondAA = posAAid
          else:
            firstAA = posAAid
            secondAA = mutAAid
          features2.append(score)
          features2.append(blosum62scaled[(firstAA, secondAA)])
          #if score > blosumCutoffsDict[blosum62[(firstAA, secondAA)]]:
          #  features.append(1)
          #else:
          #  features.append(-1)
  return snapScoreDict, snapConfDict, featureDict, feature2Dict

def loadBindingResidues(bindingResiduesFile):
    bindPosDict = {}
    with open(bindingResiduesFile, 'r') as bindFile:
        content = bindFile.readlines()
        for line in content:
            line = line.strip('\n')
            fields = line.split()
            key = fields[0]
            val = fields[1].split(',')
            bindPosDict[key] = val
    return bindPosDict

def prepareData(featureDict, feature2Dict, bindingDict):
    train = []
    train2 = []
    train_labels = []
    for protein in bindingDict:
        if protein in featureDict:
            proteinFeaturesByPos = featureDict[protein]
            proteinFeatures2ByPos = feature2Dict[protein]
            for aaPos in range(len(proteinFeaturesByPos)):
              #print(aaPos+1)
              #print(bindingDict[protein])
              if str(aaPos+1) in bindingDict[protein]:
                train_labels.append([1])
                #print("!IN!")
              else:
                train_labels.append([0])
              train.append(proteinFeaturesByPos[aaPos])
              train2.append(proteinFeatures2ByPos[aaPos])
    return train, train2, train_labels




def calc_roc(test_pred, test_labels, predCutoff = 0.4):
  tp = 0
  fp = 0
  tn = 0
  fn = 0
  for i, pred in enumerate(test_pred):
    if pred.item() > predCutoff and test_labels[i][0] == 1:
      tp = tp + 1
    elif pred.item() > predCutoff:
      fp = fp + 1
    elif test_labels[i][0] == 1:
      fn = fn + 1
    else:
      tn = tn + 1
  return tp, fp, tn, fn


if pickleFileTrain and pickleFileLabel:
  with open (pickleFileTrain, 'rb') as pft:
    train = pickle.load(pft)
  with open (pickleFileLabel, 'rb') as pfl:
    labels = pickle.load(pfl)
else:
  bindingDict = loadBindingResidues(bindingResiduesFile)
  proteinSeqDict = loadFastaFiles(fastaFolder)
  snapScoreDict, snapConfDict, featureDict, feature2Dict = loadSnapCalcFeature(snapFolder, blosum62, blosum62scaled, blosumCutoffsDict)
  train, train2, labels = prepareData(featureDict, feature2Dict, bindingDict)
  with open('train2.pickle', 'wb',) as handle:
    pickle.dump(train2, handle, protocol=pickle.HIGHEST_PROTOCOL)
  with open('labels.pickle', 'wb',) as handle:
    pickle.dump(labels, handle, protocol=pickle.HIGHEST_PROTOCOL)
  # train = with cutoff
  # train2 = raw data (blosum & SNAP2)
  train = train2

print("Finished preparing data")


D_in, H, D_out = 40, hiddenLayers, 1

model = torch.nn.Sequential(
          torch.nn.Linear(D_in, H),
          torch.nn.Sigmoid(),
          torch.nn.Linear(H, D_out),
          torch.nn.Sigmoid()
        ).to(device)

torch.save(model.state_dict(), modelPath)

weights = torch.tensor([weightNonbinding, weightBinding], device=device)
loss_fn = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)


if testmode == False:
  train_data = train # [:100000]
  train_labels = labels # [:100000]
  test_data = train[100000:]
  test_labels = labels[100000:]

  trainTensors = torch.tensor(train_data, dtype=torch.float)
  labelTensors = torch.tensor(train_labels, dtype=torch.float)
  train_and_labels = TensorDataset(trainTensors, labelTensors)
  trainloader = DataLoader(train_and_labels, batch_size=batchSize, shuffle=True)

  for t in range(epochs):
    for i, data in enumerate(trainloader):
      train_batch, labels_batch = data
      y_pred = model(train_batch)
      loss_fn.weight = weights[labels_batch.long()]
      loss = loss_fn(y_pred, labels_batch)
      print(t, loss.item())
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

  torch.save(model.state_dict(), trainedModelPath)

else:
  test_data = train
  test_labels = labels
  model.load_state_dict(torch.load(trainedModelPath))

testDataTensors = torch.tensor(test_data, dtype=torch.float)
test_pred = model(testDataTensors)
tp, fp, tn, fn = calc_roc(test_pred, test_labels, predCutoff)
finalTP = tp
finalFP = fp
finalTN = tn
finalFN = fn
mcc = (finalTP * finalTN - finalFP * finalFN)/math.sqrt((finalTP + finalFP) * (finalTP + finalFN) * (finalTN + finalFP) * (finalTN + finalFN))
prec = finalTP/(finalTP+finalFP)
recall = finalTP/(finalTP + finalFN)
print("TP: "+str(finalTP))
print("FP: "+str(finalFP))
print("TN: "+str(finalTN))
print("FN: "+str(finalFN))
print("TPR: "+str(finalTP/(finalTP+finalFN)))
print("FPR: "+str(finalFP/(finalFP+finalTN)))
print("Precision: "+str(prec))
print("F1-score: " + str(2*(prec*recall)/(prec + recall)))
print("MCC: "+str(mcc))

runtime = time.time() - timer
print("Runtime: ", runtime)
