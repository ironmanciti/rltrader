import pandas as pd
import numpy as np
from sqlalchemy import create_engine

engine = create_engine("mysql+pymysql://yjoh:1234@localhost/stockdb?charset=utf8",convert_unicode=True)

def load_chart_data(stock_code):
    conn = engine.connect()
    chart_data = pd.read_sql("select dailycandle.date, dailycandle.open, dailycandle.high, dailycandle.low,\
                dailycandle.close, dailycandle.volume, credit_ratio, foreigner_net_buy, inst_net_buy\
                from dailycandle, dailyprice where dailycandle.code ='" + stock_code +
                "' and dailyprice.code ='" + stock_code + "' and dailycandle.date = dailyprice.date \
                order by dailycandle.date", con=engine)
    chart_data.columns = ['date', 'open', 'high', 'low', 'close', 'volume',
                            'credit_ratio', 'frgn', 'inst']
    conn.close()
    return chart_data

def load_market_data(code):
    conn = engine.connect()
    market_data = pd.read_sql("select date, open, high, low, close, volume \
                             from marketcandle where code ='" + code + "'", con=engine)
    market_data.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
    conn.close()
    return market_data

def preprocess_close_volume(data, windows):
    prep_data = data
    for window in windows:
        prep_data['close_ma{}'.format(window)] = prep_data['close'].rolling(window).mean()
        prep_data['volume_ma{}'.format(window)] = (prep_data['volume'].rolling(window).mean())
    return prep_data

def preprocess_inst_frgn(data, windows):
    prep_data = data
    prep_data['inst'] = prep_data['inst'] / prep_data['inst'].sum()
    prep_data['frgn'] = prep_data['frgn'] / prep_data['frgn'].sum()
    return prep_data

def build_training_data_close_volume_ratio(prep_data, prefix, windows):
    training_data = prep_data

    training_data[prefix+'open_lastclose_ratio'] = np.zeros(len(training_data))
    training_data.loc[1:, prefix+'open_lastclose_ratio'] = \
        (training_data['open'][1:].values - training_data['close'][:-1].values) / \
        training_data['close'][:-1].values
    training_data[prefix+'high_close_ratio'] = \
        (training_data['high'].values - training_data['close'].values) / \
        training_data['close'].values
    training_data[prefix+'low_close_ratio'] = \
        (training_data['low'].values - training_data['close'].values) / \
        training_data['close'].values
    training_data[prefix+'close_lastclose_ratio'] = np.zeros(len(training_data))
    training_data.loc[1:, prefix+'close_lastclose_ratio'] = \
        (training_data['close'][1:].values - training_data['close'][:-1].values) / \
        training_data['close'][:-1].values
    training_data[prefix+'volume_lastvolume_ratio'] = np.zeros(len(training_data))
    training_data.loc[1:, prefix+'volume_lastvolume_ratio'] = \
        (training_data['volume'][1:].values - training_data['volume'][:-1].values) / \
        training_data['volume'][:-1]\
            .replace(to_replace=0, method='ffill') \
            .replace(to_replace=0, method='bfill').values

    for window in windows:
        training_data[prefix+'close_ma%d_ratio' % window] = \
            (training_data['close'] - training_data['close_ma%d' % window]) / \
            training_data['close_ma%d' % window]
        training_data[prefix+'volume_ma%d_ratio' % window] = \
            (training_data['volume'] - training_data['volume_ma%d' % window]) / \
            training_data['volume_ma%d' % window]

    return training_data
