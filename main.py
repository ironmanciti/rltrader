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
from keras import backend as K
K.clear_session()

# model save path
saved_models = []
model_dir = os.path.join(BASE_DIR, 'models/%s' % STOCK_CODE)
if not os.path.isdir(model_dir):
    os.makedirs(model_dir)
model_path = os.path.join(model_dir, 'model_%s.h5' % STOCK_CODE)

def update_delayed_reward(episode_buffer, last_reward):
    rewarded_buffer = []

    for state, action, reward, next_state, done in episode_buffer:
        if reward == 0:
            reward = last_reward
        rewarded_buffer.append((state, action, reward, next_state, done))

    return rewarded_buffer

# 강화학습 시작
if LEARNING:
    chart_data, training_data = \
            data_manager.prepare_data(STOCK_CODE, MARKET_CODE, TRAINING_START_DATE, TRAINING_END_DATE)

    env = Environment(chart_data, training_data)

    STATE_SIZE = len(training_data.columns) + 4

    replay_buffer = deque()

    agent = DQNAgent(env, STATE_SIZE)

    # initialize copy q_net --> target_net
    agent.target_model.set_weights(agent.model.get_weights())

    win_cnt = 0
    lose_cnt = 0
    for episode in range(MAX_EPISODES):
        e = 1. / (episode / 20 + 1)
        done = False
        state = env.reset()
        state = np.reshape(state, [state.shape[0], 1, STATE_SIZE])

        episode_buffer = []

        while not done:

            if np.random.rand(1) < e:
                action = random.randrange(ACTION_SIZE)
                print("random action : {}".format(action))
            else:
                predicted = np.sum(agent.model.predict(state), axis=0)
                action = np.argmax(predicted)
                print("action = {}, predict = {} / {} / {}"
                        .format(action, predicted[0], predicted[1], predicted[2]))

            if action == BUY:        # 0
                agent.num_buy  += 1   # 매수 횟수
            elif action == SELL:     # 1
                agent.num_sell += 1  # 매도 횟수
            else:
                agent.num_hold += 1  # 홀딩 횟수

            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [next_state.shape[0], 1, STATE_SIZE])

            if done or len(episode_buffer) > EPISODE_BUFFER_SIZE:

                if len(episode_buffer) > 0:
                    rewarded_buffer = update_delayed_reward(episode_buffer, reward)
                    replay_buffer.append(rewarded_buffer)
                    if len(replay_buffer) > REPLAY_MEMORY:
                        replay_buffer.popleft()

                if episode > MAX_EPISODES / 2:
                    if reward >= 1:
                        win_cnt += 1
                    elif reward <= -1:
                        lose_cnt += 1

                print("episode: {}/{}, reward = {}, episode_buffer={}, win= {}, lose= {}".
                        format(episode, MAX_EPISODES, reward, len(episode_buffer), win_cnt, lose_cnt))
                print("total buy/sell/hold: {} / {} / {}".format(agent.num_buy,agent.num_sell,agent.num_hold))
            else:
                episode_buffer.append((state, action, reward, next_state, done))

            state = next_state

        if episode % 10 == 1:
            for _ in range(10):
                agent.replay(replay_buffer, BATCH_SIZE)
            agent.target_model.set_weights(agent.model.get_weights())

    # 모델을 파일로 저장
    agent.model.save_weights(model_path, overwrite=True)

if SIMULATION:
    chart_data, training_data = \
            data_manager.prepare_data(STOCK_CODE, MARKET_CODE, SIMULATION_START_DATE, SIMULATION_END_DATE)

    env = Environment(chart_data, training_data)

    STATE_SIZE = len(training_data.columns) + 4

    agent = DQNAgent(env, STATE_SIZE)
    agent.model.load_weights(model_path)

    win_cnt = 0
    lose_cnt = 0
    done = False

    while True:

        state = env.reset()

        if env.eof == True:
            break

        while not done:
            state = np.reshape(state, [state.shape[0], 1, STATE_SIZE])
            predicted = np.sum(agent.model.predict(state), axis=0)[0]
            action = np.argmax(predicted)

            if action == BUY:        # 0
                agent.num_buy  += 1   # 매수 횟수
            elif action == SELL:     # 1
                agent.num_sell += 1  # 매도 횟수
            else:
                agent.num_hold += 1  # 홀딩 횟수

            next_state, reward, done, profitloss = env.step(action)
            next_state = np.reshape(next_state, [next_state.shape[0], 1, STATE_SIZE])

            agent.profitloss += profitloss

            state = next_state

        if reward > 0:
            win_cnt += 1
        elif reward < 0:
            lose_cnt += 1

        print("total buy/sell/hold: {}/{}/{}, win= {}, lose= {}".
                format(agent.num_buy, agent.num_sell, agent.num_hold, win_cnt, lose_cnt))
