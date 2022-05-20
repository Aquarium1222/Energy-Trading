import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from calendar import monthrange

import os
import glob
import utils.datautils as datautils
from config import Config


class EnergyDataset(Dataset):
    def __init__(self, dirpath):
        # Loading datas
        training_files = glob.glob(os.path.join(dirpath, '*.csv'))
        df_list = [pd.read_csv(training_file) for training_file in training_files]
        concat_df = pd.concat(df_list, ignore_index=True)
        month_list, day_list, hour_list, weekday_list, weekend_list = [], [], [], [], []
        for i, (_, row) in enumerate(concat_df.iterrows()):
            if i % int((len(concat_df) / 100)) == 0:
                print('\rLoading datas ({}%)'.format(int(i / (len(concat_df) / 100) + 1)), end='', flush=True)
            day, hour, weekday, weekend = datautils.break_down_time_label(row['time'])
            day_list.append(day)
            hour_list.append(hour)
            weekday_list.append(weekday)
            weekend_list.append(weekend)

        nday_list = []
        for i, day in enumerate(day_list):
            if i+1 == len(day_list) or (day_list[i] != 1 and day_list[i+1] == 1):
                start = i - day_list[i] * Config.HOUR_OF_DAY + 1
                nday_list.extend(datautils.cyclical_encoding(day_list[start:i+1], day_list[i]))
        concat_df['day'] = (np.array(nday_list) + 1) / 2
        concat_df['hour'] = (datautils.cyclical_encoding(hour_list, Config.HOUR_OF_DAY) + 1) / 2
        concat_df['weekday'] = (datautils.cyclical_encoding(weekday_list, Config.DAY_OF_WEEK) + 1) / 2
        concat_df['weekend'] = weekend_list
        concat_df[['generation', 'consumption']] = \
            datautils.minmax(concat_df[['generation', 'consumption']], True)
        self.concat_df = concat_df.drop(columns=['time'])

        # Making datas
        # (N, L, C)
        # N: batch size
        # L: length of sequence
        # C: number of feature
        self.incoming_len = Config.INCOMING_DAYS * Config.HOUR_OF_DAY
        self.output_len = Config.OUTPUT_DAYS * Config.HOUR_OF_DAY
        self.total_len = int((len(day_list) - (self.incoming_len + self.output_len)) / Config.HOUR_OF_DAY)
        print('\nTotal len: {}'.format(self.total_len))

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        s = idx * Config.HOUR_OF_DAY
        e = s + self.incoming_len
        x_df = self.concat_df.iloc[s:e]
        y_df = self.concat_df.iloc[e:e + self.output_len]
        x = np.array([
            x_df['day'],
            x_df['hour'],
            x_df['weekday'],
            x_df['generation'],
            x_df['consumption']
        ]).transpose()
        y = np.array([
            y_df['day'],
            y_df['hour'],
            y_df['weekday'],
            y_df['generation'],
            y_df['consumption']
        ]).transpose()
        return x, y
