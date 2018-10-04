import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_chart_data(stock_code):
    engine = create_engine("mysql+pymysql://yjoh:1234@localhost/stockdb?charset=utf8",convert_unicode=True)
    conn = engine.connect()

    chart_data = pd.read_sql("select date, open, high, low, close, volume \
                             from dailycandle where code ='" + stock_code + "'", con=engine)
    chart_data.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
    conn.close()
    return chart_data


def preprocess(chart_data):
    prep_data = chart_data
    windows = [5, 10, 20, 60, 120]
    for window in windows:
        prep_data['close_ma{}'.format(window)] = prep_data['close'].rolling(window).mean()
        prep_data['volume_ma{}'.format(window)] = (
            prep_data['volume'].rolling(window).mean())
    return prep_data


def build_training_data(prep_data):
    training_data = prep_data

    training_data['open_lastclose_ratio'] = np.zeros(len(training_data))
    training_data.loc[1:, 'open_lastclose_ratio'] = \
        (training_data['open'][1:].values - training_data['close'][:-1].values) / \
        training_data['close'][:-1].values
    training_data['high_close_ratio'] = \
        (training_data['high'].values - training_data['close'].values) / \
        training_data['close'].values
    training_data['low_close_ratio'] = \
        (training_data['low'].values - training_data['close'].values) / \
        training_data['close'].values
    training_data['close_lastclose_ratio'] = np.zeros(len(training_data))
    training_data.loc[1:, 'close_lastclose_ratio'] = \
        (training_data['close'][1:].values - training_data['close'][:-1].values) / \
        training_data['close'][:-1].values
    training_data['volume_lastvolume_ratio'] = np.zeros(len(training_data))
    training_data.loc[1:, 'volume_lastvolume_ratio'] = \
        (training_data['volume'][1:].values - training_data['volume'][:-1].values) / \
        training_data['volume'][:-1]\
            .replace(to_replace=0, method='ffill') \
            .replace(to_replace=0, method='bfill').values

    windows = [5, 10, 20, 60, 120]
    for window in windows:
        training_data['close_ma%d_ratio' % window] = \
            (training_data['close'] - training_data['close_ma%d' % window]) / \
            training_data['close_ma%d' % window]
        training_data['volume_ma%d_ratio' % window] = \
            (training_data['volume'] - training_data['volume_ma%d' % window]) / \
            training_data['volume_ma%d' % window]

    return training_data


# chart_data = pd.read_csv(fpath, encoding='CP949', thousands=',', engine='python')
