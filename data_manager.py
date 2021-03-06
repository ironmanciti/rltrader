import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from settings import *

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
    prep_data['inst'] = prep_data['inst'].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0)
    prep_data['frgn'] = prep_data['frgn'].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0)
    return prep_data

# 전처리된 data 에 stock/market+전일종가비율/저가종가비율/종가전일종가비율/거래량전일거래량비율 column 생성
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
            .replace(to_replace=0, method='bfill').values     # 거래량이 0 이면 무한대 비울이 나오므로 다른 값으로 채워줌

    for window in windows:
        training_data[prefix+'close_ma%d_ratio' % window] = \
            (training_data['close'] - training_data['close_ma%d' % window]) / \
            training_data['close_ma%d' % window]              # 이동평균종가비율
        training_data[prefix+'volume_ma%d_ratio' % window] = \
            (training_data['volume'] - training_data['volume_ma%d' % window]) / \
            training_data['volume_ma%d' % window]             # 이동평균거래량비율

    return training_data

def prepare_data(stock_code, market_code, start_date, end_date):
    # 종목 데이터 준비
    chart_data = load_chart_data(stock_code)
    prep_data = preprocess_close_volume(chart_data, WINDOWS)
    data = build_training_data_close_volume_ratio(prep_data,'stock', WINDOWS)

    # data1 = preprocess_inst_frgn(data1, WINDOWS)

     # market 데이터 준비
    # market_data = load_market_data(market_code)
    # prep_data = preprocess_close_volume(market_data, WINDOWS)
    # data2 = build_training_data_close_volume_ratio(prep_data,'market', WINDOWS)
    # 종목 + market data merge
    # data = pd.merge(data, data2, on='date', suffixes=('', '_y'))
    # 기간 필터링
    data = data[(data['date'] >= start_date) & (data['date'] <= end_date)]
    data = data.dropna()
    # 차트 데이터 분리
    features_chart_data = ['date', 'open', 'high', 'low', 'close', 'volume']
    chart_data = data[features_chart_data]
    # 추가 feature 데이터 분리
    # drop_features = ['date', 'open', 'high', 'low', 'volume',
    #                  'open_y', 'high_y', 'low_y', 'close_y', 'volume_y']
    drop_features = ['date', 'open', 'high', 'low', 'volume', 'credit_ratio', 'frgn', 'inst']
    for window in WINDOWS:
        drop_features.append('close_ma{}'.format(window))
        drop_features.append('volume_ma{}'.format(window))
        # drop_features.append('close_ma{}_y'.format(window))
        # drop_features.append('volume_ma{}_y'.format(window))

    training_data = data.drop(columns=drop_features)
    # scaling
    training_data['close'] -= training_data['close'].min()
    training_data['close'] /= training_data['close'].max()

    return chart_data, training_data
