import argparse
#import matplotlib.pyplot as plt # pylint: disable=import-error
import os
import ntpath
import re
import NeuralNetwork # pylint: disable=import-error
import torch
import pickle
from Bio.SubsMat import MatrixInfo as matrices

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
pickleFileTrain = args.pickleTrain
pickleFileLabel = args.pickleLabel

blosum62 = matrices.blosum62
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

def loadFastaFiles(fastaFolder):
  proteinSeqDict = {}
  for filename in os.listdir(fastaFolder):
    with open(os.path.join(fastaFolder, filename), 'r') as fastaFile:
      header = fastaFile.readline()
      header = header.strip('\n')
      sequence = fastaFile.readline()
      proteinSeqDict[header[1:]] = sequence
  return proteinSeqDict

def loadSnapCalcFeature(snapFolder, blosum62, blosumCutoffsDict):
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
          features2.append(blosum62[(firstAA, secondAA)])
          if score > blosumCutoffsDict[blosum62[(firstAA, secondAA)]]:
            features.append(1)
          else:
            features.append(-1)
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

if pickleFileTrain and pickleFileLabel:
  with open (pickleFileTrain, 'rb') as pft:
    train = pickle.load(pft)
  with open (pickleFileLabel, 'rb') as pfl:
    train_labels = pickle.load(pfl)
else:
  bindingDict = loadBindingResidues(bindingResiduesFile)
  proteinSeqDict = loadFastaFiles(fastaFolder)
  snapScoreDict, snapConfDict, featureDict, feature2Dict = loadSnapCalcFeature(snapFolder, blosum62, blosumCutoffsDict)
  train, train2, train_labels = prepareData(featureDict, feature2Dict, bindingDict)
  with open('train.pickle', 'wb',) as handle:
    pickle.dump(train, handle, protocol=pickle.HIGHEST_PROTOCOL)
  with open('train2.pickle', 'wb',) as handle:
    pickle.dump(train2, handle, protocol=pickle.HIGHEST_PROTOCOL)
  with open('labels.pickle', 'wb',) as handle:
    pickle.dump(train_labels, handle, protocol=pickle.HIGHEST_PROTOCOL)
  train = train2


print("Finished preparing data")
NN = NeuralNetwork.Neural_Network()
trainTensors = torch.tensor(train, dtype=torch.float)
labelTensors = torch.tensor(train_labels, dtype=torch.float)


#for x in range(len(train)):  # trains the NN 1,000 times
#  i = torch.tensor([train[x]], dtype=torch.float)
#  o = torch.tensor(train_labels[x], dtype=torch.float)
#  print ("#" + str(x) + " Loss: " + str(torch.mean((o - NN(i))**2).detach().item()))  # mean sum squared loss
#  NN.train(i, o)


for i in range(100):  # trains the NN 1,000 times
  print ("#" + str(i) + " Loss: " + str(torch.mean((labelTensors - NN(trainTensors))**2).detach().item()))  # mean sum quared loss
  NN.train(trainTensors, labelTensors)

NN.saveWeights(NN)
NN.predict(bindVals=train[54], nonBindVals=train[55])
