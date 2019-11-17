import logging
import os
import pandas as pd
import settings
import data_manager
from policy_learner import PolicyLearner

WINDOWS = [5, 10, 20, 60, 120]

def prepare_data(stock_code, market_code, start_date, end_date):
    # 종목 데이터 준비
    chart_data = data_manager.load_chart_data(stock_code)
    prep_data = data_manager.preprocess_close_volume(chart_data, WINDOWS)
    data1 = data_manager.build_training_data_close_volume_ratio(prep_data,'stock', WINDOWS)
    data1 = data_manager.preprocess_inst_frgn(data1, WINDOWS)
     # market 데이터 준비
    market_data = data_manager.load_market_data(market_code)
    prep_data = data_manager.preprocess_close_volume(market_data, WINDOWS)
    data2 = data_manager.build_training_data_close_volume_ratio(prep_data,'market', WINDOWS)
    # 종목 + market data merge
    data = pd.merge(data1, data2, on='date', suffixes=('', '_y'))
    # 기간 필터링
    data = data[(data['date'] >= start_date) & (data['date'] <= end_date)]
    data = data.dropna()
    # 차트 데이터 분리
    features_chart_data = ['date', 'open', 'high', 'low', 'close', 'volume']
    chart_data = data[features_chart_data]
    # 추가 feature 데이터 분리
    drop_features = ['date', 'open', 'high', 'low', 'close', 'volume',
                     'open_y', 'high_y', 'low_y', 'close_y', 'volume_y']
    for window in WINDOWS:
        drop_features.append('close_ma{}'.format(window))
        drop_features.append('volume_ma{}'.format(window))
        drop_features.append('close_ma{}_y'.format(window))
        drop_features.append('volume_ma{}_y'.format(window))

    training_data = data.drop(columns=drop_features)

    return chart_data, training_data

if __name__ == '__main__':

    STOCK_CODE = '035420'  # '005380'-현대차, '005930'-삼성전자, '051910'-LG화학, '035420'-NAVER, '030200'-KT, '000660'-SK하이닉스
    MARKET_CODE = '001'    # '001'-KOSPI
    LEARNING = True
    SIMULATION = True
    INITIAL_BALANCE = int(('10,000,000').replace(',',''))
    MIN_TRADING_UNIT = 1  #50
    MAX_TRADING_UNIT = 2 #100
    NUM_EPOCHS = 800
    MAX_MEMORY = 60
    START_EPSILON = 0.5   # 50%
    DELAYED_REWARD_THRESHOLD = 0.05   # 5%
    LEARNING_RATE = 0.01
    DISCOUNT_FACTOR = 0      # p.106 과거로 갈수록 지연보상을 약하게 할 경우
    TRAINING_START_DATE   = '2016-01-01'
    TRAINING_END_DATE     = '2016-12-31'
    SIMULATION_START_DATE = '2017-01-01'
    SIMULATION_END_DATE   = '2017-12-31'

    # 로그 기록
    log_dir = os.path.join(settings.BASE_DIR, 'logs/%s' % STOCK_CODE)
    timestr = settings.get_time_str()
    if not os.path.exists('logs/%s' % STOCK_CODE):
        os.makedirs('logs/%s' % STOCK_CODE)
    file_handler = logging.FileHandler(filename=os.path.join(
        log_dir, "%s_%s.log" % (STOCK_CODE, timestr)), encoding='utf-8')
    stream_handler = logging.StreamHandler()  # send logging output to stdout
    file_handler.setLevel(logging.DEBUG)
    stream_handler.setLevel(logging.INFO)
    logging.basicConfig(format="%(message)s",
                        handlers=[file_handler, stream_handler], level=logging.DEBUG)

    # 강화학습 시작
    if LEARNING:
        chart_data, data = prepare_data(STOCK_CODE, MARKET_CODE, TRAINING_START_DATE, TRAINING_END_DATE)

        policy_learner = PolicyLearner(
            stock_code=STOCK_CODE, chart_data=chart_data, training_data=data,
            min_trading_unit=MIN_TRADING_UNIT, max_trading_unit=MAX_TRADING_UNIT,
            delayed_reward_threshold=DELAYED_REWARD_THRESHOLD, lr=LEARNING_RATE)
            
        policy_learner.fit(balance=INITIAL_BALANCE, num_epoches=NUM_EPOCHS,max_memory=MAX_MEMORY,
                        discount_factor=DISCOUNT_FACTOR, start_epsilon=START_EPSILON, learning=LEARNING_RATE)

        # 정책 신경망을 파일로 저장
        model_dir = os.path.join(settings.BASE_DIR, 'models/%s' % STOCK_CODE)
        if not os.path.isdir(model_dir):
            os.makedirs(model_dir)
        #model_path = os.path.join(model_dir, 'model_%s.h5' % timestr)
        model_path = os.path.join(model_dir, 'model_%s.h5' % STOCK_CODE)
        policy_learner.policy_network.save_model(model_path)

    if SIMULATION:
        chart_data, data = prepare_data(STOCK_CODE, MARKET_CODE, SIMULATION_START_DATE, SIMULATION_END_DATE)

        policy_learner = PolicyLearner(
            stock_code=STOCK_CODE, chart_data=chart_data, training_data=data,
            min_trading_unit=MIN_TRADING_UNIT, max_trading_unit=MAX_TRADING_UNIT)

        policy_learner.trade(balance=INITIAL_BALANCE, num_epoches=NUM_EPOCHS,max_memory=MAX_MEMORY,
                           discount_factor=DISCOUNT_FACTOR, start_epsilon=START_EPSILON,
                             model_path=os.path.join(
                                 settings.BASE_DIR,
                                 'models/{}/model_{}.h5'.format(STOCK_CODE, STOCK_CODE)))
        #                        'models/{}/model_{}.h5'.format(STOCK_CODE, model_ver)))
