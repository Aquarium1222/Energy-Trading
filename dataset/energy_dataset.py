import numpy as np
import pandas as pd
from torch.utils.data import Dataset

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
        day_list, hour_list, weekday_list = [], [], []
        for i, (_, row) in enumerate(concat_df.iterrows()):
            if i % int((len(concat_df) / 100)) == 0:
                print('\rLoading datas ({}%)'.format(int(i / (len(concat_df) / 100)+1)), end='', flush=True)
            day, hour, weekday = datautils.break_down_time_label(row['time'])
            day_list.append(day)
            hour_list.append(hour)
            weekday_list.append(weekday)
        print('')
        generation_list = concat_df['generation']
        consumption_list = concat_df['consumption']

        # Making datas
        incoming_len = Config.INCOMING_DAYS * Config.HOUR_OF_DAY
        output_len = Config.OUTPUT_DAYS * Config.HOUR_OF_DAY
        total_len = len(day_list) - (incoming_len + output_len)
        x, y = [], []
        for s in range(total_len):
            if s % int((total_len / 100)) == 0:
                print('\rMaking datas ({}%)'.format(int(s / (total_len / 100)+1)), end='', flush=True)
            e = s + incoming_len
            x = np.array([
                day_list[s:s+incoming_len],
                hour_list[s:s+incoming_len],
                weekday_list[s:s+incoming_len],
                generation_list[s:s+incoming_len],
                consumption_list[s:s+incoming_len]
            ])
            y = np.array([
                generation_list[e:e+output_len],
                consumption_list[e:e+output_len]
            ])
        print('')
        self.x, self.y = x.transpose(), y.transpose()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x, self.y
