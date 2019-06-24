import argparse

parser = argparse.ArgumentParser(description='Convert allParams to csv')
parser.add_argument('--paramsFile', help = "Path to file containing results of param optimization", type = str)
args = parser.parse_args()

paramsFile = args.paramsFile
newLine = ""
fout = open(paramsFile.split(".")[0]+".csv", "w+")
fout.write("BatchSize,HiddenNodes,WeightNonbinding,WeightBinding,LearningRate,Epochs,Momentum,BestCutoff,TruePositives,FalsePositives,TrueNegatives,FalseNegatives,MCC\n")
with open(paramsFile, "r") as fin:
  for line in fin:
    if line.startswith("----"):
      newLine = newLine[:-1]
      newLine = newLine+"\n"
      fout.write(newLine)
      newLine = ""
      continue
    if ":" not in line:
      continue
    val = line.split(": ")[1].rstrip()
    newLine = newLine + val + ","
fout.close()
      