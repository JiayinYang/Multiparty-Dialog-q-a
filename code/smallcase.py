import torch
import numpy as np
import torch.nn as nn
partition = [1,2,3,4]
labels = [5,6,7,8]


class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, list_IDs, labels):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        X = ID
        y = self.labels[index]

        return X, y

# Parameters
params = {'batch_size': 1,
          'shuffle': True}
max_epochs = 100



# Generators
training_set = Dataset(partition, labels)
training_generator = torch.utils.data.DataLoader(training_set, **params)


for epoch in range(max_epochs):
    for local_X, local_y in training_generator:
    	print(len(local_X))