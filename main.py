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

def update_delayed_reward(episode_buffer, last_reward):
    rewarded_buffer = []

    for state, action, reward, next_state, done in episode_buffer:
        reward = last_reward         # 마지막 reward 로 앞선 action 의 reward 모두 change
        rewarded_buffer.append((state, action, reward, next_state, done))

    return rewarded_buffer

model_dir = os.path.join(BASE_DIR, 'models/%s' % STOCK_CODE)
if not os.path.isdir(model_dir):
    os.makedirs(model_dir)

# 강화학습 시작
if LEARNING:
    chart_data, training_data = \
            data_manager.prepare_data(STOCK_CODE, MARKET_CODE, TRAINING_START_DATE, TRAINING_END_DATE)

    env = Environment(chart_data, training_data)

    STATE_SIZE = len(training_data.columns) + 4

    replay_buffer = deque()

    agent = DQNAgent(env, STATE_SIZE)

    # 이전에 저장한 weight 에서 학습 시작
    if MODEL_NAME:
        print("=================================================")
        print("CONTINUE_LEARNING from model = : ", MODEL_NAME)
        print("=================================================")
        print("          ")
        model_path = os.path.join(model_dir, MODEL_NAME)
        agent.model.load_weights(model_path)

    # initialize copy q_net --> target_net
    agent.target_model.set_weights(agent.model.get_weights())

    yymmdd = datetime.datetime.now().strftime("%Y-%m-%d")

    win_cnt = 0
    lose_cnt = 0
    e = 1

    for episode in range(MAX_EPISODES):
        if e > 0.05:
            e = 1. / (episode / 10 + 1)
        done = False
        state = env.reset("LEARNING")
        state = np.reshape(state, [1, state.shape[0], STATE_SIZE])

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
            next_state = np.reshape(next_state, [1, next_state.shape[0], STATE_SIZE])

            if done or len(episode_buffer) > EPISODE_BUFFER_SIZE:

                if len(episode_buffer) > 0:
                    rewarded_buffer = update_delayed_reward(episode_buffer, reward)
                    replay_buffer.append(rewarded_buffer)
                    if len(replay_buffer) > REPLAY_MEMORY:
                        replay_buffer.popleft()

                if episode > MAX_EPISODES / 2:
                    if done and reward >= 5:
                        win_cnt += 1
                    elif done and reward <= -5:
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

        if episode % 100 == 1:
            # 모델을 파일로 중간 저장
            model_path = os.path.join(model_dir, 'model_{}_{}_{}.h5'.format(STOCK_CODE, yymmdd, episode))
            agent.model.save_weights(model_path, overwrite=True)
            print("episode: {}/{}, model saved : {}".format(episode, MAX_EPISODES, model_path))

    # 마지막 모델 저장
    model_path = os.path.join(model_dir, 'model_{}_{}_final.h5'.format(STOCK_CODE, yymmdd))
    agent.model.save_weights(model_path, overwrite=True)
    print("model saved : ", model_path)

if SIMULATION:

    simulation_model_path = None

    if not simulation_model_path:
        simulation_model_path = model_path

    print("Simulation begins...model = : ", simulation_model_path)

    agent.model.load_weights(simulation_model_path)

    chart_data, training_data = \
            data_manager.prepare_data(STOCK_CODE, MARKET_CODE, SIMULATION_START_DATE, SIMULATION_END_DATE)

    env = Environment(chart_data, training_data)

    STATE_SIZE = len(training_data.columns) + 4

    agent = DQNAgent(env, STATE_SIZE)

    win_cnt = 0
    lose_cnt = 0
    done = False

    state = env.reset("SIMULATION")
    state = np.reshape(state, [1, state.shape[0], STATE_SIZE])

    while True:

        if env.eof == True:
            break

        while not done:
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

            next_state, reward, done, profitloss = env.step(action)
            next_state = np.reshape(next_state, [1, next_state.shape[0], STATE_SIZE])

            if done:
                agent.profitloss += profitloss

            state = next_state

        if reward >= 2:
            win_cnt += 1
        elif reward <= -2:
            lose_cnt += 1

        print("total buy/sell/hold: {}/{}/{}, win= {}, lose= {}, profit_loss = {}".
                format(agent.num_buy, agent.num_sell, agent.num_hold, win_cnt, lose_cnt, agent.profitloss))

        next_state, reward, done, profitloss = env.step(action)
        next_state = np.reshape(next_state, [1, next_state.shape[0], STATE_SIZE])

        state = next_state
