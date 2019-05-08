import argparse
import matplotlib.pyplot as plt # pylint: disable=import-error
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

def createFeatureHist(featureDict):
  featureSumFrequencies = []
  for featureArrays in featureDict.values():
    for featureArray in featureArrays:
      featureSumFrequencies.append(featureArray.count(1))
  plt.hist(x=featureSumFrequencies, bins = range(0,20), align = 'left')
  plt.xticks(range(0,20))
  plt.xticks(rotation=90)
  plt.xlabel('Feature Sum')
  plt.ylabel('Frequency')
  plt.title('Histogram of feature sum frequencies')
  plt.savefig("featureHistogram.pdf")

proteinSeqDict = loadFastaFiles(fastaFolder)
snapScoreDict, snapConfDict, featureDict = loadSnapCalcFeature(snapFolder, blosum62, blosumCutoffsDict)
createFeatureHist(featureDict)