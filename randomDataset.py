import random

# number of training positions
trainsize = 100000
# number of testing positions
testsize = 20000

ls = [0] * 92 + [1] * 8

def TrainSet(ls):
  trainlabels = []
  trainfeatures = []
  for i in range(trainsize):
    trainlabels.append(random.choice(ls))
    feat = []
    for j in range(40):
      feat.append(random.randint(-100, 100))
    trainfeatures.append(feat)

  return trainlabels, trainfeatures

def TestSet(ls):
  testlabels = []
  testfeatures = []
  for i in range(testsize):
    testlabels.append(random.choice(ls))
    feat = []
    for j in range(40):
      feat.append(random.randint(-100, 100))
    testfeatures.append(feat)

  return testlabels, testfeatures

trainlabels, trainfeatures = TrainSet(ls)
testlabels, testfeatures = TestSet(ls)

with open()




