import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from config import Config


def break_down_time_label(time_label):
    dt = datetime.strptime(time_label, '%Y-%m-%d %H:%M:%S')
    return dt.day, dt.hour, dt.weekday()


def make_train_test_indices(dataset):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(Config.TEST_SIZE * dataset_size))
    np.random.shuffle(indices)
    return indices[split:], indices[:split]


def make_dataloader(dataset, indices, batch_size):
    sampler = SubsetRandomSampler(indices)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)
