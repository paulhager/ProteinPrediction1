import argparse
#import matplotlib.pyplot as plt # pylint: disable=import-error
import os
import ntpath
import re
import NeuralNetwork # pylint: disable=import-error
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pickle
import math
from Bio.SubsMat import MatrixInfo as matrices
import time
import copy
torch.manual_seed(0)

timer = time.time()
batchSize = 1000
hiddenLayers = 200
weightNonbinding = 0.4
weightBinding = 0.6
learning_rate = 3e-3
epochs = 2
device = torch.device('cpu')
crossValidation = False
predCutoff = 0.4
momentum=0.9
#device = torch.device('cuda')
blosumScalar = 1
modelPath = "initialModel"

parser = argparse.ArgumentParser(description='Load and analyse protein binding site data')
parser.add_argument('--fastaFolder', help = "Path to folder containing fasta files", type = str)
parser.add_argument('--snapFolder', help = "Path to folder containing snap files", type = str)
parser.add_argument('--bindingResidues', help = "Path to binding residues files", type = str)
parser.add_argument('--pickleTrain', help = "Path to pickle file with train data structures", type=str)
parser.add_argument('--pickleLabel', help = "Path to pickle file with label data structures", type=str)

args = parser.parse_args()

fastaFolder = args.fastaFolder
snapFolder = args.snapFolder
bindingResiduesFile = args.bindingResidues
pickleFileTrain = 'C:\\Users\\Thomas\\Documents\\Uni_masters\\ProteinPrediction1\\train2.pickle'
pickleFileLabel = 'C:\\Users\\Thomas\\Documents\\Uni_masters\\ProteinPrediction1\\labels.pickle'


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

def createFeatureHist(featureDict):
  featureSumFrequencies = []
  for featureArrays in featureDict.values():
    for featureArray in featureArrays:
      featureSumFrequencies.append(featureArray.count(1))
  f = plt.figure(1)
  plt.hist(x=featureSumFrequencies, bins = range(0,20), align = 'left')
  plt.xticks(range(0,20))
  plt.xticks(rotation=90)
  plt.xlabel('Feature Sum')
  plt.ylabel('Frequency')
  plt.title('Histogram of feature sum frequencies')
  f.savefig("featureHistogram.pdf")

def pos_hist(proteinSeqDict, bindingDict):
    matched = matchmaker(proteinSeqDict, bindingDict)
    positions = []
    for i in matched:
        positions = positions + i[1]
    all_pos = []
    for j in positions:
        if j != '':
            all_pos.append(int(j))
    g = plt.figure(2)
    plt.hist(all_pos, 100)
    plt.title('Distribution of binding residues between sequence positions')
    plt.xlabel('Position in sequence')
    plt.ylabel('Number of binding residues')
    g.savefig('positionHistogram.pdf')

def matchmaker(proteinSeqDict, bindingDict):
    matched = []
    for i in proteinSeqDict:
        if i in bindingDict:
                matched.append([proteinSeqDict[i], bindingDict[i]])
    return matched

def bindCount(proteinSeqDict, bindingDict):
    matched = matchmaker(proteinSeqDict, bindingDict)
    y = []
    x = []
    for i in matched:
        x.append(len(i[1]))
        y.append(len(i[0]))
    h = plt.figure(3)
    plt.plot(y, x, 'ro')
    plt.title('Number of binding residues per sequence length')
    plt.xlabel('Sequence length')
    plt.ylabel('Number of binding residues')
    h.savefig('seqLenBindNumDotplot.pdf')

def tot_lens(proteinSeqDict, bindingDict):
    matched = matchmaker(proteinSeqDict, bindingDict)
    total_len = [0,0]
    for i in range(len(matched)):
        total_len[0] = total_len[0] + len(matched[i][0])
        total_len[1] = total_len[1] + len(matched[i][1])
    nonbind = total_len[0] - total_len[1]
    k = plt.figure(4)
    labels = 'binding_residues', 'non-binding residues'
    plt.pie([total_len[1], nonbind], labels=labels,startangle=90, autopct='%1.1f%%')
    plt.axis('equal')
    k.savefig('lengthPie.pdf')

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
  with open('train.pickle', 'wb',) as handle:
    pickle.dump(train, handle, protocol=pickle.HIGHEST_PROTOCOL)
  with open('train2.pickle', 'wb',) as handle:
    pickle.dump(train2, handle, protocol=pickle.HIGHEST_PROTOCOL)
  with open('labels.pickle', 'wb',) as handle:
    pickle.dump(labels, handle, protocol=pickle.HIGHEST_PROTOCOL)
  # train = with cutoff
  # train2 = raw data (blosum & SNAP2)
  train = train2

print("Finished preparing data")
NN = NeuralNetwork.Neural_Network()

D_in, H, D_out = 40, hiddenLayers, 1

model = torch.nn.Sequential(
          torch.nn.Conv1d(D_in, 10, kernel_size = 9, stride = 1, padding = 4),
          torch.nn.Sigmoid(),
          torch.nn.Conv1d(10, 1, kernel_size = 5, stride = 1, padding = 2),
          torch.nn.Sigmoid(),
        ).to(device)

torch.save(model.state_dict(), modelPath)

weights = torch.tensor([weightNonbinding, weightBinding], device=device)
loss_fn = torch.nn.BCELoss(reduction = 'mean')
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

if crossValidation:
  splitSize = math.floor(len(train)/5)
  split1_data = train[:splitSize]
  split1_labels = labels[:splitSize]
  split2_data = train[splitSize:splitSize*2]
  split2_labels = labels[splitSize:splitSize*2]
  split3_data = train[splitSize*2:splitSize*3]
  split3_labels = labels[splitSize*2:splitSize*3]
  split4_data = train[splitSize*3:splitSize*4]
  split4_labels = labels[splitSize*3:splitSize*4]
  split5_data = train[splitSize*4:]
  split5_labels = labels[splitSize*4:]

  allSplitsData = [split1_data, split2_data, split3_data, split4_data, split5_data]
  allSplitsLabels = [split1_labels, split2_labels, split3_labels, split4_labels, split5_labels]

  allTP = 0
  allFP = 0
  allTN = 0
  allFN = 0
  for x in range(5):
    model = torch.nn.Sequential(
          torch.nn.Linear(D_in, H),
          torch.nn.Sigmoid(),
          torch.nn.Linear(H, D_out),
          torch.nn.Sigmoid()
        ).to(device)
    model.load_state_dict(torch.load(modelPath))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # Cross-validation
    currentTrainData = []
    currentLabelsData = []
    for y in range(5):
      if x != y:
        currentTrainData.extend(allSplitsData[y])
        currentLabelsData.extend(allSplitsLabels[y])
    currentTestData = allSplitsData[x]
    currentTestLabels = allSplitsLabels[x]
    # initialize for training
    trainTensors = torch.tensor(currentTrainData, dtype=torch.float, device=device)
    labelTensors = torch.tensor(currentLabelsData, dtype=torch.float, device=device)
    train_and_labels = TensorDataset(trainTensors, labelTensors)
    trainloader = DataLoader(train_and_labels, batch_size=batchSize, shuffle=True)
    # train
    for t in range(epochs):
      for i, data in enumerate(trainloader):
        train_batch, labels_batch = data
        y_pred = model(train_batch)
        loss_fn.weight = weights[labels_batch.long()]
        loss = loss_fn(y_pred, labels_batch)
        print(x+1, t, loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # evaluate
    testDataTensors = torch.tensor(currentTestData, dtype=torch.float)
    test_pred = model(testDataTensors)
    tp, fp, tn, fn = calc_roc(test_pred, currentTestLabels, predCutoff)
    allTP = allTP + tp
    allFP = allFP + fp
    allTN = allTN + tn
    allFN = allFN + fn
  # output final stats:
  print("TP: "+str(allTP))
  print("FP: "+str(allFP))
  print("TN: "+str(allTN))
  print("FN: "+str(allFN))
  print("TPR: "+str(allTP/(allTP+allFN)))
  print("FPR: "+str(allFP/(allFP+allTN)))
  print("Precision: "+str(allTP/(allTP+allFP)))
  print("MCC: "+str(((allTP*allTN)-(allFP*allFN))/(math.sqrt((allTP+allFP)*(allTP+allFN)*(allTN+allFP)*(allTN+allFN)))))

else:
  train_data = train[:100000]
  train_labels = labels[:100000]
  test_data = train[100000:]
  test_labels = labels[100000:]

  trainTensors = torch.tensor(train_data, dtype=torch.float)
  labelTensors = torch.tensor(train_labels, dtype=torch.float)
  train_and_labels = TensorDataset(trainTensors, labelTensors)
  trainloader = DataLoader(train_and_labels, batch_size=batchSize, shuffle=True)

  for t in range(epochs):
    for i, data in enumerate(trainloader):
      train_batch, labels_batch = data
      y_pred = model(train_batch.unsqueeze(2))
      loss_fn.weight = weights[labels_batch.long()]
      loss = loss_fn(y_pred, labels_batch)
      print(t, loss.item())
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

  testDataTensors = torch.tensor(test_data, dtype=torch.float)
  test_pred = model(testDataTensors.unsqueeze(2))
  bestMCC = -1
  bestCutoff = 0
  for posCutoff in range(100):
    posCutoff = posCutoff / 100
    tp, fp, tn, fn = calc_roc(test_pred, test_labels, posCutoff)
    denominator = math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    if denominator != 0:
      mcc = ((tp*tn)-(fp*fn))/denominator
      if mcc > bestMCC:
        bestMCC = mcc
        bestCutoff = posCutoff
        finalTP = tp
        finalFP = fp
        finalTN = tn
        finalFN = fn

  print("Best cutoff: "+str(bestCutoff))
  print("TP: "+str(finalTP))
  print("FP: "+str(finalFP))
  print("TN: "+str(finalTN))
  print("FN: "+str(finalFN))
  print("TPR: "+str(finalTP/(finalTP+finalFN)))
  print("FPR: "+str(finalFP/(finalFP+finalTN)))
  print("Precision: "+str(finalTP/(finalTP+finalFP)))
  print("MCC: "+str(bestMCC))

runtime = time.time() - timer
print("Runtime: ", runtime)
#for t in range(500):
#  y_pred = model(trainTensors)
#  loss = loss_fn(y_pred, labelTensors)
#  print(t, loss.item())
#  optimizer.zero_grad()
#  loss.backward()
#  optimizer.step()

#bindTest = torch.tensor((train[54]), dtype=torch.float)
#nonBindTest = torch.tensor((train[55]), dtype=torch.float)
#print ("Predicted data binding site: ")
#print ("Input (scaled): \n" + str(bindTest))
#print ("Output: \n" + str(model(bindTest)))
#print ("-------")
#print ("Predicted data non-binding site: ")
#print ("Input (scaled): \n" + str(nonBindTest))
#print ("Output: \n" + str(model(nonBindTest)))

#for x in range(len(train)):  # trains the NN 1,000 times
#  i = torch.tensor([train[x]], dtype=torch.float)
#  o = torch.tensor(train_labels[x], dtype=torch.float)
#  print ("#" + str(x) + " Loss: " + str(torch.mean((o - NN(i))**2).detach().item()))  # mean sum squared loss
#  NN.train(i, o)


#for i in range(400):  # trains the NN 1,000 times
#  print ("#" + str(i) + " Loss: " + str(torch.mean((labelTensors - NN(trainTensors))**2).detach().item()))  # mean sum quared loss
#  NN.train(trainTensors, labelTensors)

#NN.saveWeights(NN)
#NN.predict(bindVals=train[54], nonBindVals=train[55])
