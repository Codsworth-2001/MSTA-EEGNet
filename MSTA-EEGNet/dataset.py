import torch
from torch.utils.data import Dataset as tdataset
import numpy as np
import scipy
import os

dataset_size_1 = 0
base_PATH_1 = "your_dataset_path"


def data_processing(sample_size, name, input_value):
    x_data = []
    dir = []
    dataset_size = 0
    if name == 'SEED':
        base_PATH = base_PATH_1
        dataset_size = dataset_size_1
        all_folders = [folder for folder in os.listdir(base_PATH) if os.path.isdir(os.path.join(base_PATH, folder))]
        if input_value == "all":
            dir = all_folders
        else:
            dir = [folder for folder in all_folders if folder.startswith(f"{input_value}_")]

    for dirname in dir:

        fname = 'your path here'
        f = scipy.io.loadmat(fname)
        x = f[]
       x_data = x[0, :]

    data = np.reshape(x_data, [62, -1, sample_size])
    data = np.transpose(data, (1, 0, 2))
    return data


def label_processing(sample_size, name, input_value):
    x_label = []
    if name == 'SEED':
        base_PATH = base_PATH_1
        dataset_size = dataset_size_1
        all_folders = [folder for folder in os.listdir(base_PATH) if os.path.isdir(os.path.join(base_PATH, folder))]
        if input_value == "all":
            dir = all_folders
        else:
            dir = [folder for folder in all_folders if folder.startswith(f"{input_value}_")]
    for dirname in dir:
        fname ='your path here'
        f = scipy.io.loadmat(fname)
        x = f[]
        x_label = x[1, :]
    label = np.reshape(x_label, [-1])

    return label


class MyDataset(tdataset):
    def __init__(self, sample_size, num_patch, name, input_value, transform=None):
        super().__init__()
        self.transform = transform
        dataset = data_processing(sample_size * num_patch, name, input_value)
        raw_label = label_processing(sample_size * num_patch, name, input_value)
        n, l, d = dataset.shape
        raw_label = raw_label - 1
        dataset = dataset.reshape(-1, l, sample_size)

        self.dataset = np.squeeze(dataset)
        self.label = raw_label[0:len(dataset)]

    def __getitem__(self, index):
        x, y = torch.FloatTensor(np.array(self.dataset[index])), torch.LongTensor(np.array(self.label[index]))
        if self.transform is not None:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.dataset)
