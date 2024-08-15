from torch.utils.data import Dataset
import numpy as np
import torch
import os


class AvatarDataset(Dataset):

    def __init__(self, root_dir, train):
        super(AvatarDataset, self).__init__()
        if train:
            self.name = 'trainset'
            self.inputs = np.load(os.path.join(root_dir, 'inputs_train.npy'))
            self.outputs = np.load(os.path.join(root_dir, 'outputs_train.npy'))
        else:
            self.name = 'testset'
            self.inputs = np.load(os.path.join(root_dir, 'inputs_test.npy'))
            self.outputs = np.load(os.path.join(root_dir, 'outputs_test.npy'))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input = self.inputs[idx]
        output = self.outputs[idx]
        return torch.from_numpy(input), torch.from_numpy(output)
