
import argparse
import os
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pickle
import math
from Bio.SubsMat import MatrixInfo as matrices
import time
import copy
import randomDataset
import matplotlib.pyplot as plt
import random
import statistics

timer = time.time()
batchSize = 1000
hiddenLayers = 200
weightNonbinding = 0.4
weightBinding = 0.6
learning_rate = 3e-3
epochs = 100
device = torch.device('cpu')
crossValidation = False
predCutoff = 0.4
momentum=0.9
#device = torch.device('cuda')
blosumScalar = 1

torch.manual_seed(13)

trainedModelPath = 'trainedModel.pickle'

parser = argparse.ArgumentParser(description='Load and analyse protein binding site data')
parser.add_argument('--fastaFolder', help = "Path to folder containing fasta files", type = str)
parser.add_argument('--snapFolder', help = "Path to folder containing snap files", type = str)
parser.add_argument('--bindingResidues', help = "Path to binding residues files", type = str)
parser.add_argument('--pickleTrain', help = "Path to pickle file with train data structures", type=str)
parser.add_argument('--pickleLabel', help = "Path to pickle file with label data structures", type=str)
parser.add_argument('--trainedModel', help='Path to pre-trained model', type=str)
parser.add_argument('--randomData', help='creates a random Dataset', type = bool)

args = parser.parse_args()

fastaFolder = args.fastaFolder
snapFolder = args.snapFolder
bindingResiduesFile = args.bindingResidues
pickleFileTrain = args.pickleTrain
pickleFileLabel = args.pickleLabel
trainedmodel = args.trainedModel
randomData = args.randomData
if trainedmodel != None:
    testmode = True
else:
    testmode = False
if randomData == None:
  randomMode = False
else:
  randomMode = randomData
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


def validate(test_data,target, model, epoch):
    with torch.no_grad():
        target = torch.Tensor(target)
        testDataTensors = torch.tensor(test_data, dtype=torch.float)
        test_pred = model(testDataTensors)
        loss = torch.sqrt(torch.nn.functional.binary_cross_entropy(test_pred, target))
        print('Validation loss after epoch {} is {:.2}'.format(epoch, loss))
        #tp, fp, tn, fn = calc_roc(test_pred, test_labels, predCutoff)
    return loss


def distributionPlots(train):
  # unscale Blosum first
  g = plt.figure(2)
  snapscores = []
  blosumscores = []
  for val in train:
    for i in range(len(val)):
      if i % 2 == 0:
        snapscores.append(val[i])
      else:
        blosumscores.append(val[i])
  plt.hist(x=snapscores, bins=8)
  plt.title('Distribution single of Snap-scores')
  plt.xlabel('Snap-score')
  plt.ylabel('Occurences in the Dataset')
  plt.tight_layout()
  g.savefig('DistributionofSnapscores.png')
  h = plt.figure(3)
  bins = [-4, -3, -2, -1, 0, 1, 2, 3]
  height = []
  for i in bins:
    z = blosumscores.count(i)
    height.append(z)
  plt.bar(bins, height)
  plt.title('Distribution of BLOSUM62-scores')
  plt.xlabel('BLOSUM62-score')
  plt.ylabel('Occurences in the Dataset')
  plt.tight_layout()
  h.savefig('DistributionofBlosumscores.png')

def bootstrapper(resultpath):
  labs = []
  predictions = []
  with open(resultpath, 'r') as f:
    for line in f:
      line = line.split('\t')
      labs.append(line[0])
      predictions.append(line[1])
  f.close()
  allmccs = []
  allprec = []
  allrecall = []
  allf1 = []
  errorcount = 0
  for j in range(1000):
    new_lab = []
    new_pred = []
    for i in range(30000):
      num = random.randint(0, len(labs) - 1)
      new_lab.append([float(labs[num])])
      new_pred.append(float(predictions[num]))
    new_pred = torch.FloatTensor(new_pred)
    tp, fp, tn, fn = calc_roc(new_pred, new_lab)
    x = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if x != 0:
      mcc = (tp * tn - fp * fn) / x
      allmccs.append(mcc)
    else:
      errorcount += 1
    if tp + fp != 0:
      prec = tp / (tp + fp)
      allprec.append(prec)
    if tp + fn != 0:
      recall = tp / (tp + fn)
      allrecall.append(recall)
    if prec + recall != 0:
      f1 = 2 * (prec * recall) / (prec + recall)
      allf1.append(f1)
  stderrmcc = statistics.stdev(allmccs)
  stderrprec = statistics.stdev(allprec)
  stderrrecall = statistics.stdev(allrecall)
  stderrf1 = statistics.stdev(allf1)
  print('bootstrapping errors:', errorcount)
  return stderrmcc, stderrprec, stderrrecall, stderrf1

def randomPredictor(testresultsPath):
  labs = []
  randpreds = []
  tp = 0
  tn = 0
  fp = 0
  fn = 0
  ls = [0] * 92 + [1]*8
  with open(testresultsPath, 'r') as f:
    for line in f:
      line = line.split('\t')
      labs.append(int(line[0]))
  f.close()
  for i in range(len(labs)):
    randpreds.append(random.choice(ls))
  for j in range(len(labs)):
    if labs[j] == randpreds[j]:
      if labs[j] == 0:
        tn += 1
      else:
        tp += 1
    else:
      if labs[j] == 0:
        fp += 1
      else:
        fn += 1

  print('Randompredictions')
  print('TP:', tp)
  print('FP:', fp)
  print('TN:', tn)
  print('FN:', fn)
  x = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
  if x != 0:
    mcc = (tp * tn - fp * fn) / x
    print(mcc)
  print('End randompredictions')

def calc_roc(test_pred, test_labels, predCutoff = 0.4):
  tp = 0
  fp = 0
  tn = 0
  fn = 0
  with open('testresults.txt', 'w+') as results:

    for i, pred in enumerate(test_pred):
      if pred.item() > predCutoff and test_labels[i][0] == 1:
        results.write(str(test_labels[i][0]) + '\t' + str(pred.item()) + '\t' + 'TP' + '\n')
        tp = tp + 1
      elif pred.item() > predCutoff:
        results.write(str(test_labels[i][0]) + '\t' + str(pred.item()) + '\t' + 'FP' + '\n')
        fp = fp + 1
      elif test_labels[i][0] == 1:
        results.write(str(test_labels[i][0]) + '\t' + str(pred.item()) + '\t' + 'FN' + '\n')
        fn = fn + 1
      else:
        results.write(str(test_labels[i][0]) + '\t' + str(pred.item()) + '\t' + 'TN' + '\n')
        tn = tn + 1
    results.close()
  return tp, fp, tn, fn

if pickleFileTrain and pickleFileLabel and not randomMode:
  with open (pickleFileTrain, 'rb') as pft:
    train = pickle.load(pft)
  with open (pickleFileLabel, 'rb') as pfl:
    labels = pickle.load(pfl)
elif not randomMode:
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
else:
  rand = randomDataset.random_Dataset()
  labels, train = rand.TrainSet()

print("Finished preparing data")


D_in, H, D_out = len(train[0]), hiddenLayers, 1

model = torch.nn.Sequential(
          torch.nn.Linear(D_in, H),
          torch.nn.Sigmoid(),
          torch.nn.Linear(H, D_out),
          torch.nn.Sigmoid()
        ).to(device)


weights = torch.tensor([weightNonbinding, weightBinding], device=device)
loss_fn = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)


if testmode == False:
  if not randomMode:
    train_data = train[:100000]
    train_labels = labels[:100000]
    test_data = train[100000:]
    test_labels = labels[100000:]
    # for random Dataset, ignore otherwise
    # rand = randomDataset.random_Dataset()
    # test_labels, test_data = rand.TestSet()

  else:
    # rand = randomDataset.random_Dataset()
    train_data = train
    train_labels = labels
    test_labels, test_data = rand.TestSet()

  trainTensors = torch.tensor(train_data, dtype=torch.float)
  labelTensors = torch.tensor(train_labels, dtype=torch.float)
  train_and_labels = TensorDataset(trainTensors, labelTensors)
  trainloader = DataLoader(train_and_labels, batch_size=batchSize, shuffle=True)

  for t in range(epochs):
    loss_list = []
    for i, data in enumerate(trainloader):
      train_batch, labels_batch = data
      y_pred = model(train_batch)
      loss_fn.weight = weights[labels_batch.long()]
      loss = loss_fn(y_pred, labels_batch)
      loss_list.append(loss)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

    if t % 10 == 0:
      train_loss = sum(loss_list)/len(loss_list)
      print('Training loss after epoch {} is {:.2}'.format(t, train_loss))
      val_loss = validate(test_data,test_labels, model, t)
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

# distributionPlots(train)

# randomPredictor('C:\\Users\\thoma\\Documents\\Uni\\6.Semester\\protpred1\\testresults.txt')


stderrmcc, stderrprec, stderrrecall, stderrf1 = bootstrapper('C:\\Users\\thoma\\Documents\\Uni\\6.Semester\\proteinPrediction\\excercise\\testresults.txt')
print('stderrMCC:', stderrmcc)
print('stderr Precision:', stderrprec)
print('stderr Recall:', stderrrecall)
print('Stderr F1:', stderrf1)

with open('teststatistics.txt', 'w+') as stats:
  stats.write("TP:" + '\t' + str(finalTP) + '\n')
  print("TP: "+str(finalTP))
  stats.write("FP:" + '\t' + str(finalFP) + '\n')
  print("FP: "+str(finalFP))
  stats.write("TN:" + '\t' + str(finalTN) + '\n')
  print("TN: "+str(finalTN))
  stats.write("FN:" + '\t' + str(finalFN) + '\n')
  print("FN: "+str(finalFN))

  if (finalFP+finalTN) != 0:
    stats.write("FPR:" + '\t' + str(finalFP / (finalFP + finalTN)) + '\n')
    print("FPR: "+str(finalFP/(finalFP+finalTN)))
  else:
    stats.write("FPR cannot be calculated (Division by zero)" + '\n')
    print("FPR cannot be calculated (Division by zero)")
  prec = 0
  if finalTP+finalFP != 0:
    prec = finalTP/(finalTP+finalFP)
    stats.write("Precision:" + '\t' + str(prec) + '\n')
    print("Precision: " + str(prec))
  else:
    stats.write("Precision cannot be calculated (Division by zero)" + '\n')
    print("Precision cannot be calculated (Division by zero)")
  recall = 0
  if (finalTP + finalFN) != 0:
    recall = finalTP/(finalTP + finalFN)
    stats.write("Recall/TPR:" + '\t' + str(recall) + '\n')
    print("Recall/TPR: " + str(recall))
  else:
    stats.write("Recall/TPR cannot be calculated (Division by zero)" + '\n')
    print("Recall/TPR cannot be calculated (Division by zero)")
  if prec + recall != 0:
    stats.write("F1-score:" + '\t' + str(2*(prec*recall)/(prec + recall)) + '\n')
    print("F1-score: " + str(2*(prec*recall)/(prec + recall)))
  else:
    stats.write("F1-score cannot be calculated (Division by zero)" + '\n')
    print("F1-score cannot be calculated (Divion by zero)")
  x = math.sqrt((finalTP + finalFP) * (finalTP + finalFN) * (finalTN + finalFP) * (finalTN + finalFN))
  if x != 0:
    mcc = (finalTP * finalTN - finalFP * finalFN)/x
    stats.write("MCC:" + '\t' + str(mcc) + '\n')
    print("MCC: "+str(mcc))
  else:
    stats.write("MCC cannot be calculated (Division by zero)" + '\n')
    print("MCC can not be calculated (Division by Zero)")
  stats.close()

  runtime = time.time() - timer
  print("Runtime: ", runtime)
