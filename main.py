import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import utils.datautils as datautils
from config import Config


# You should not modify this part.
def config():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--consumption", default="./sample_data/consumption.csv", help="input the consumption data path")
    parser.add_argument("--generation", default="./sample_data/generation.csv", help="input the generation data path")
    parser.add_argument("--bidresult", default="./sample_data/bidresult.csv", help="input the bids result path")
    parser.add_argument("--output", default="output.csv", help="output the bids path")

    return parser.parse_args()


def output(path, data):
    df = pd.DataFrame(data, columns=["time", "action", "target_price", "target_volume"])
    df.to_csv(path, index=False)


if __name__ == "__main__":
    args = config()

    # Load model
    model = torch.load(Config.CHECKPOINT_DIR + Config.MODEL_NAME)

    # Prepare input data
    con_df = pd.read_csv(args.consumption)
    gen_df = pd.read_csv(args.generation)
    df = con_df.copy(True)
    df['generation'] = gen_df['generation']
    day_list, hour_list, weekday_list, weekend_list = [], [], [], []
    for i, row in df.iterrows():
        day, hour, weekday, weekend = datautils.break_down_time_label(row['time'])
        day_list.append(day)
        hour_list.append(hour)
        weekday_list.append(weekday)
        weekend_list.append(weekend)
    df['day'] = (datautils.cyclical_encoding(day_list, 31) + 1) / 2
    df['hour'] = (datautils.cyclical_encoding(hour_list, Config.HOUR_OF_DAY) + 1) / 2
    df['weekday'] = (datautils.cyclical_encoding(weekday_list, Config.DAY_OF_WEEK) + 1) / 2
    df['weekend'] = weekend_list
    df[['generation', 'consumption']] = \
        datautils.minmax(df[['generation', 'consumption']], True)
    df = df.drop(columns=['time'])
    data = torch.FloatTensor(
        np.expand_dims(df[['day', 'hour', 'weekday', 'generation', 'consumption']].to_numpy(), axis=1))
    mock_y = torch.FloatTensor(np.random.rand(24, 1, 5))

    # Predict
    relu = torch.nn.ReLU()
    pred = relu(model(data, mock_y).detach().cpu())
    pred = pred.squeeze()[:, -2:]
    pred = np.round_(datautils.inverse_minmax(pred), decimals=2)

    # Sell
    bid_df = pd.read_csv(args.bidresult)
    if bid_df.empty:
        buy_price = Config.BUY_INIT_PRICE
        sell_price = Config.SELL_INIT_PRICE
    else:
        buy_df = bid_df[bid_df['action'] == 'buy']
        sell_df = bid_df[bid_df['action'] == 'sell']
        buy_price = Config.BUY_INIT_PRICE if buy_df[buy_df['trade_price'] >= 0].empty \
            else np.mean(buy_df[buy_df['trade_price'] >= 0]['trade_price'])

        sell_price, deal_count = 0, 0
        for i, row in sell_df.iterrows():
            # deal_ratio <= 1
            trade_volume = float(row['trade_volume'])
            trade_price = float(row['trade_price'])
            target_volume = float(row['target_volume'])
            target_price = float(row['target_price'])

            deal_ratio = trade_volume / target_volume
            if trade_price != -1:
                deal_count += 1
                sell_price += trade_price * (deal_ratio / Config.THRESHOLD)
        sell_price = Config.SELL_INIT_PRICE if deal_count == 0 else sell_price / deal_count

        buy_price = Config.BUY_INIT_PRICE if buy_price > Config.BUY_INIT_PRICE else buy_price
        sell_price = Config.SELL_INIT_PRICE if sell_price > Config.SELL_INIT_PRICE else sell_price

    # Making output
    [[con_error, gen_error]] = datautils.inverse_minmax([[Config.CON_LOSS, Config.GEN_LOSS]])

    data = []
    dt = datetime.strptime(con_df['time'].iloc[-1], '%Y-%m-%d %H:%M:%S')
    for each_pred in pred:
        dt = dt + timedelta(hours=1)
        gen = each_pred[0] - gen_error
        con = each_pred[1] + con_error
        diff = gen - con
        if diff == 0:
            continue
        elif diff < 0:
            action = 'buy'
        else:
            action = 'sell'
        data.append([
            datetime.strftime(dt, '%Y-%m-%d %H:%M:%S'),
            action,
            buy_price if action == 'buy' else sell_price,
            abs(diff)
        ])
    output(args.output, data)
