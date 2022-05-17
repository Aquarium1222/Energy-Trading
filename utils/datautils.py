import os.path
import pickle
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.preprocessing import MinMaxScaler

from config import Config


def break_down_time_label(time_label):
    dt = datetime.strptime(time_label, '%Y-%m-%d %H:%M:%S')
    return dt.day, dt.hour, dt.weekday(), dt.weekday() >= 5


def cyclical_encoding(data_list, scope):
    return np.sin(np.array(data_list) * (2 * np.pi / scope))


def make_train_test_indices(dataset):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(Config.TEST_SIZE * dataset_size))
    np.random.shuffle(indices)
    return indices[split:], indices[:split]


def make_dataloader(dataset, indices, batch_size):
    sampler = SubsetRandomSampler(indices)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)


def minmax(df, output_dir=''):
    scaler = MinMaxScaler()
    tran_df = scaler.fit_transform(df)
    if output_dir != '':
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        dump_pickle(scaler, os.path.join(output_dir, Config.SCALER_NAME))
    return tran_df


def dump_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj
