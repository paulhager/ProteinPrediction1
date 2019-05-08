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
  for filename in os.listdir(snapFolder):
    proteinID = os.path.basename(filename)[:-6]
    snapScoreDict[proteinID] = []
    with open(os.path.join(snapFolder, filename), 'r') as snapFile:
      for line in snapFile:
        m = re.search(r"^(.*?)\s", line)
        if m:
          posAAid = m.group(1)
          if aaOrder.index(posAAid[0]) == aaIndex:
            aaIndex += 1
            snapScoreDict[proteinID].append(None)
            continue
          m = re.search(r"score = (.*)", line)
          if m:
            score = m.group(1)
            snapScoreDict[proteinID].append(score)
            if aaIndex != 19:
              aaIndex += 1
            else:
              posAAscores = []
              aaIndex = 0
  return snapScoreDict


def calcFeatureLists(proteinSeqDict, snapScoreDict):
  print(len(proteinSeqDict))
  print(len(snapScoreDict))
  print(len(snapScoreDict['P12733']))
  print(snapScoreDict['P12733'][4])
  print(snapScoreDict['P12733'][5])

  featureLists = []
  return featureLists


proteinSeqDict = loadFastaFiles(fastaFolder)
snapScoreDict = loadSnapFiles(snapFolder)
featureLists = calcFeatureLists(proteinSeqDict, snapScoreDict)