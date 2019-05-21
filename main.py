import argparse
#import matplotlib.pyplot as plt # pylint: disable=import-error
import os
import ntpath
import re
from Bio.SubsMat import MatrixInfo as matrices

parser = argparse.ArgumentParser(description='Load and analyse protein binding site data')
parser.add_argument('--fastaFolder', help = "Path to folder containing fasta files", type = str)
parser.add_argument('--snapFolder', help = "Path to folder containing snap files", type = str)
parser.add_argument('--bindingResidues', help = "Path to binding residues files", type = str)

args = parser.parse_args()

fastaFolder = args.fastaFolder
snapFolder = args.snapFolder
bindingResiduesFile = args.bindingResidues

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
  for filename in os.listdir(snapFolder):
    aaIndex = 0
    proteinID = os.path.basename(filename)[:-6]
    snapScoreDict[proteinID] = []
    snapConfDict[proteinID] = []
    featureDict[proteinID] = []
    posAAscores = []
    posAAconf = []
    features = []
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
          posAAscores.append(None)
          posAAconf.append(None)
          features.append(None)
        if score[-1] == '%':
          score = score[:-1]
        score = int(score)
        if len(posAAscores) > len(posAAconf):
          posAAconf.append(score)
          if aaIndex < 19:
            aaIndex += 1
            if(aaIndex == posAAindex == 19):
                posAAscores.append(None)
                posAAconf.append(None)
                features.append(None)
                snapScoreDict[proteinID].append(posAAscores)
                snapConfDict[proteinID].append(posAAconf)
                featureDict[proteinID].append(features)
                posAAscores = []
                posAAconf = []
                features = []
                aaIndex = 0
          else:
            snapScoreDict[proteinID].append(posAAscores)
            snapConfDict[proteinID].append(posAAconf)
            featureDict[proteinID].append(features)
            posAAscores = []
            posAAconf = []
            features = []
            aaIndex = 0
        else:
          posAAscores.append(score)
          if posAAindex < mutAAindex:
            firstAA = mutAAid
            secondAA = posAAid
          else:
            firstAA = posAAid
            secondAA = mutAAid
          if score > blosumCutoffsDict[blosum62[(firstAA, secondAA)]]:
            features.append(1)
          else:
            features.append(0)
  return snapScoreDict, snapConfDict, featureDict

def loadBindingResidues(bindingResiduesFile):
    bindPosDict = {}
    with open(bindingResiduesFile, 'r') as bindFile:
        content = bindFile.readlines()
        for line in content:
            line = line.strip('\n')
            key = line[:6]
            val = line[7:].split(',')
            bindPosDict[key] = val
    return bindPosDict

def prepareData(featureDict, bindingDict):
    train = []
    train_labels = []
    for protein in bindingDict:
        if protein in featureDict:
            proteinFeaturesByPos = featureDict[protein]
            for aaPos in range(len(proteinFeaturesByPos)):
                train_labels.append(protein+"_"+str(aaPos))
                train.append(proteinFeaturesByPos[aaPos])
    return train, train_labels


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

proteinSeqDict = loadFastaFiles(fastaFolder)
snapScoreDict, snapConfDict, featureDict = loadSnapCalcFeature(snapFolder, blosum62, blosumCutoffsDict)
bindingDict = loadBindingResidues(bindingResiduesFile)
train, train_labels = prepareData(featureDict, bindingDict)
print(train[1])
print(train[2])
print(train[3])
print(train[4])
print(train_labels[1])
print(train_labels[2])
print(train_labels[3])
print(train_labels[4])
#createFeatureHist(featureDict)
#pos_hist(proteinSeqDict, bindingDict)
#bindCount(proteinSeqDict, bindingDict)
#tot_lens(proteinSeqDict, bindingDict)