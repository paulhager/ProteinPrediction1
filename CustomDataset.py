from torch.utils import data
import torch
class Dataset(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, train_data, train_labels):
        'Initialization'
        self.train = train_data
        self.labels = train_labels

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.train)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        X = torch.tensor(self.train[index]).float()
        y = torch.tensor(self.labels[index]).float()
        return X, y