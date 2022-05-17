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
            datautils.minmax(concat_df[['generation', 'consumption']], Config.SCALER_DIR)
        concat_df = concat_df.drop(columns=['time'])
        print('')

        # Making datas
        incoming_len = Config.INCOMING_DAYS * Config.HOUR_OF_DAY
        output_len = Config.OUTPUT_DAYS * Config.HOUR_OF_DAY
        total_len = len(day_list) - (incoming_len + output_len)
        self.x_list, self.y_list = [], []
        for s in range(0, total_len, 24):
            if s % int((total_len / 100)) == 0:
                print('\rMaking datas ({}%)'.format(int(s / (total_len / 100)+1)), end='', flush=True)
            e = s + incoming_len
            x_df = concat_df.iloc[s:e]
            y_df = concat_df.iloc[e:e+output_len]
            x = np.array([
                x_df['day'],
                x_df['hour'],
                x_df['weekday'],
                x_df['generation'],
                x_df['consumption']
            ]).transpose()
            y = np.array([
                y_df['generation'],
                y_df['consumption']
            ]).transpose()
            self.x_list.append(x)
            self.y_list.append(y)
        print('')

    def __len__(self):
        return len(self.x_list)

    def __getitem__(self, idx):
        return self.x_list[idx], self.y_list[idx]
