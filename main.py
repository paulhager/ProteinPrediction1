import argparse
#import matplotlib.pyplot as plt # pylint: disable=import-error
import os
import ntpath
import re

parser = argparse.ArgumentParser(description='Load and analyse protein binding site data')
parser.add_argument('--fastaFolder', help = "Path to folder containing fasta files", type = str)
parser.add_argument('--snapFolder', help = "Path to folder containing snap files", type = str)
parser.add_argument('--bindingResidues', help = "Path to binding residues files", type = str)

args = parser.parse_args()

fastaFolder = args.fastaFolder
snapFolder = args.snapFolder
bindingResiduesFile = args.bindingResidues

def loadFastaFiles(fastaFolder):
  proteinSeqDict = {}
  for filename in os.listdir(fastaFolder):
    with open(os.path.join(fastaFolder, filename), 'r') as fastaFile:
      header = fastaFile.readline()
      sequence = fastaFile.readline()
      proteinSeqDict[header[1:]] = sequence
  return proteinSeqDict

def loadSnapFiles(snapFolder):
  aaOrder = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
  aaIndex = 0
  snapScoreDict = {}
  snapConfDict = {}
  for filename in os.listdir(snapFolder):
    proteinID = os.path.basename(filename)[:-6]
    snapScoreDict[proteinID] = []
    snapConfDict[proteinID] = []
    posAAscores = []
    posAAconf = []
    with open(os.path.join(snapFolder, filename), 'r') as snapFile:
      for line in snapFile:
        posAAid = line[0]
        if aaOrder.index(posAAid[0]) == aaIndex:
          aaIndex += 1
          posAAscores.append(None)
          posAAconf.append(None)
        splitLine = line.split()
        score = splitLine[len(splitLine)-1]
        if len(posAAscores) > len(posAAconf):
          posAAconf.append(score)
          if aaIndex < 19:
            aaIndex += 1
          else:
            snapScoreDict[proteinID].append(posAAscores)
            snapConfDict[proteinID].append(posAAconf)
            posAAscores = []
            posAAconf = []
            aaIndex = 0
        else:
          posAAscores.append(score)
  return snapScoreDict


def calcFeatureLists(proteinSeqDict, snapScoreDict):
  print(len(proteinSeqDict))
  print(len(snapScoreDict))
  print(len(snapScoreDict['P12733']))

  featureLists = []
  return featureLists


proteinSeqDict = loadFastaFiles(fastaFolder)
snapScoreDict = loadSnapFiles(snapFolder)
featureLists = calcFeatureLists(proteinSeqDict, snapScoreDict)