import logging
import os
import random
import pandas as pd
import numpy as np
from collections import deque
from settings import *
import data_manager
from agent import DQNAgent
from environment import Environment
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

def update_replay_buffer(replay_buffer, episode_buffer, reward):

    episode_rewarded = []

    for state, action, next_state, done in episode_buffer:
        episode_rewarded.append((state, action, reward, next_state, done))

    replay_buffer.append(episode_rewarded)

    if len(replay_buffer) > REPLAY_MEMORY:
        replay_buffer.popleft()


# 강화학습 시작
chart_data, training_data = \
        prepare_data(STOCK_CODE, MARKET_CODE, TRAINING_START_DATE, TRAINING_END_DATE)

env = Environment(training_data)

STATE_SIZE = len(training_data.columns) + 4

replay_buffer = deque()

agent = DQNAgent(env, STATE_SIZE)

# initialize copy q_net --> target_net
agent.target_model.set_weights(agent.model.get_weights())

last_saved_model = 'No model saved'
saved_models = []

if LEARNING:
    win_cnt = 0
    lose_cnt = 0
    for episode in range(MAX_EPISODES):
        e = 1. / (episode / 10 + 1)
        done = False
        state = env.reset()
        state = np.reshape(state, [1, 1, STATE_SIZE])

        episode_buffer = []

        while not done:
            if np.random.rand(1) < e:
                action = random.randrange(ACTION_SIZE)
            else:
                action = np.argmax(agent.model.predict(state)[0])

            next_state, reward, done = env.step(action)
            next_state = np.reshape(next_state, [1, 1, STATE_SIZE])

            if done:
                update_replay_buffer(replay_buffer, episode_buffer, reward)

                if episode > MAX_EPISODES / 2:
                    if reward >= 1:
                        win_cnt += 1
                    elif reward <= -1:
                        lose_cnt += 1

                print("episode: {}/{}, reward = {}, episode_buffer={}, win= {}, lose= {}".
                        format(episode, MAX_EPISODES, reward, len(episode_buffer), win_cnt, lose_cnt))
            else:
                episode_buffer.append((state, action, next_state, done))

            state = next_state

        if episode % 10 == 1:
            for _ in range(50):
                agent.replay(replay_buffer, BATCH_SIZE)
            agent.target_model.set_weights(agent.model.get_weights())

    #
    # policy_learner = PolicyLearner(
    #     stock_code=STOCK_CODE, chart_data=chart_data, training_data=training_data,
    #     min_trading_unit=MIN_TRADING_UNIT, max_trading_unit=MAX_TRADING_UNIT,
    #     delayed_reward_threshold=DELAYED_REWARD_THRESHOLD, lr=LEARNING_RATE)
    #
    # policy_learner.fit(balance=INITIAL_BALANCE, num_epoches=NUM_EPOCHS,max_memory=MAX_MEMORY,
    #                 discount_factor=DISCOUNT_FACTOR, start_epsilon=START_EPSILON, learning=LEARNING_RATE)
    #
    # # 정책 신경망을 파일로 저장
    # model_dir = os.path.join(settings.BASE_DIR, 'models/%s' % STOCK_CODE)
    # if not os.path.isdir(model_dir):
    #     os.makedirs(model_dir)
    # #model_path = os.path.join(model_dir, 'model_%s.h5' % timestr)
    # model_path = os.path.join(model_dir, 'model_%s.h5' % STOCK_CODE)
    # policy_learner.policy_network.save_model(model_path)

# if  SIMULATION:
#
#     policy_learner = PolicyLearner(
#         stock_code=STOCK_CODE, chart_data=chart_data, training_data=training_data,
#         min_trading_unit=MIN_TRADING_UNIT, max_trading_unit=MAX_TRADING_UNIT)
#
#     policy_learner.trade(balance=INITIAL_BALANCE, num_epoches=NUM_EPOCHS,max_memory=MAX_MEMORY,
#                        discount_factor=DISCOUNT_FACTOR, start_epsilon=START_EPSILON,
#                          model_path=os.path.join(
#                              settings.BASE_DIR,
#                              'models/{}/model_{}.h5'.format(STOCK_CODE, STOCK_CODE)))
#     #                        'models/{}/model_{}.h5'.format(STOCK_CODE, model_ver)))
