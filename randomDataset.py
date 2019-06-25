import random

class random_Dataset():
  def __init__(self, ):
    super(random_Dataset, self).__init__()
    # number of training positions
    self.trainsize = 100000
    # number of testing positions
    self.testsize = 20000

    self.ls = [0] * 92 + [1] * 8

  def TrainSet(self):
    trainlabels = []
    trainfeatures = []
    for i in range(self.trainsize):
      trainlabels.append([random.choice(self.ls)])
      feat = []
      for j in range(40):
        feat.append(random.randint(-100, 100))
      trainfeatures.append(feat)

    return trainlabels, trainfeatures

  def TestSet(self):
    testlabels = []
    testfeatures = []
    for i in range(self.testsize):
      testlabels.append([random.choice(self.ls)])
      feat = []
      for j in range(40):
        feat.append(random.randint(-100, 100))
      testfeatures.append(feat)

    return testlabels, testfeatures

# trainlabels, trainfeatures = randomDataset.TrainSet()
# testlabels, testfeatures = randomDataset.TestSet()
#
#
#


