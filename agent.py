import numpy as np
from settings import *
import random
from keras.models import Sequential
from keras.layers import Activation, Dense, LSTM, BatchNormalization
from keras.optimizers import Adam

class DQNAgent:

    def __init__(self, env, state_size):
        # Environment 객체
        self.env = env # 현재 주식 가격을 가져오기 위해 환경 참조
        self.gamma = GAMMA
        self.epsilon = EPSILON
        self.epsilon_decay = EPSILON_DECAY
        self.learning_rate = LEARNING_RATE
        self.state_size = state_size

        self.model        = self._build_model()
        self.target_model = self._build_model()

        # # 최소 매매 단위, 최대 매매 단위, 지연보상 임계치
        # self.min_trading_unit = min_trading_unit  # 최소 단일 거래 단위
        # self.max_trading_unit = max_trading_unit  # 최대 단일 거래 단위
        # self.delayed_reward_threshold = delayed_reward_threshold  # 지연보상 임계치 (손익률이 threshold 를 넘으면 지연보상 발생)
        #
        # # Agent 클래스의 속성
        # self.initial_balance = 0  # 초기 자본금 (투자 시작 시점의 보유 현금)
        # self.balance = 0  # 현재 현금 잔고
        # self.num_stocks = 0  # 현재 보유 주식 수
        # self.portfolio_value = 0  # balance + num_stocks * {현재 주식 가격} --> 포트폴리오 가치
        # self.base_portfolio_value = 0  # 직전 학습 시점의 PV (기준 포트폴리오)
        # self.base_stock_price = 0      # 직전 학습 시점의 주가
        self.num_buy = 0  # 매수 횟수
        self.num_sell = 0  # 매도 횟수
        self.num_hold = 0  # 홀딩 횟수
        self.profitloss = 0  # 누적 손익
        # self.immediate_reward = 0  # 즉시 보상 (행동을 수행한 시점에 수익발행 +1, 아니면 -1)
        # self.buy_charge = 0   # 매수 수수료
        # self.sell_charge = 0  # 매도 수수료
        # self.sell_tax  = 0    # 거래세
        #
        # # Agent 클래스의 상태
        # self.ratio_hold = 0  # 주식 보유 비율 (현재보유하고있는 주식수/최대로 보유할수 있는 주식수)
        # self.ratio_portfolio_value = 0  # 포트폴리오 가치 비율 (현재 PV/직전지연보상시점의 PV)
    #

    def _build_model(self):
        model = Sequential()
        model.add(LSTM(128, input_shape=(None, self.state_size),
                return_sequences=False, dropout=0.2))
        # model.add(BatchNormalization())
        # model.add(LSTM(64, return_sequences=False, dropout=0.2))
        model.add(BatchNormalization())
        model.add(Dense(32))
        model.add(Dense(ACTION_SIZE, activation='linear'))

        model.compile(optimizer=Adam(lr=LEARNING_RATE), loss='mse')
        return model

    def replay(self, replay_buffer, batch_size):

        if len(replay_buffer) < batch_size:
            return

        minibatch = random.sample(replay_buffer, batch_size)

        for  episode_rewarded in minibatch:
            y = np.zeros((len(episode_rewarded), ACTION_SIZE))

            for (state, action, reward, next_state, done) in episode_rewarded:

                state = state.reshape(state.shape[1], state.shape[0], state.shape[2])

                target_f = self.target_model.predict(state)

                if done:
                    target_f[:, action] = reward
                else:
                   target_f[:, action] = (reward + self.gamma *
                               np.amax(self.target_model.predict(next_state)[:, action]))

                self.model.fit(state, target_f, epochs=1, verbose=0)

    # def reset(self):   # class 속성 초기화 (매 epoch 마다)
    #     self.balance = self.initial_balance
    #     self.num_stocks = 0
    #     self.portfolio_value = self.initial_balance
    #     self.base_portfolio_value = self.initial_balance
    #     self.num_buy = 0
    #     self.num_sell = 0
    #     self.num_hold = 0
    #     self.immediate_reward = 0
    #     self.buy_charge = 0
    #     self.sell_charge = 0
    #     self.sell_tax  = 0
    #     self.ratio_hold = 0
    #     self.ratio_portfolio_value = 0
    #
    # def set_balance(self, balance):
    #     self.initial_balance = balance     # 초기 자본금 설정
    #
    # def get_states(self):                 # agent 의 상태 반환
    #     self.ratio_hold = self.num_stocks / int(
    #         self.portfolio_value / self.environment.get_price())     # 보유주식수/(PV/현재주가)
    #     self.ratio_portfolio_value = self.portfolio_value / self.base_portfolio_value     # PV/기준PV
    #     return (
    #         self.ratio_hold,              # 주식보유비율 - 0: 주식 없음, 1: 최대보유
    #         self.ratio_portfolio_value    # 포트폴리오 가치 비율 - < 1 : 손실, 1 > : 수익발생
    #     )

    # def decide_action(self, policy_network, sample, epsilon):
    #     # confidence = 0.
    #     # 탐험 결정
    #     if np.random.rand() < epsilon:
    #         exploration = True
    #         action = np.random.randint(self.NUM_ACTIONS)  # 무작위로 행동 결정
    #     else:
    #         exploration = False
    #         probs = policy_network.predict(sample)  # 각 행동에 대한 확률, 지금까지 훈련시킨 청책망으로 다음 action 결정
    #         action = np.argmax(probs)
    #         # confidence = probs[action]
    #
    #     # return action, confidence, exploration
    #     return action, exploration     # 0:매수, 1:매도, 2:HOLD / False/True(탐험여부)
    #
    # def validate_action(self, action):
    #     validity = True
    #     if action == Agent.ACTION_BUY:  # 매수의 경우
    #         # 적어도 1주를 살 수 있는지 확인
    #         if self.balance < self.environment.get_price() * (
    #             1 + self.TRADING_CHARGE) * self.min_trading_unit:
    #             validity = False
    #     elif action == Agent.ACTION_SELL:   # 매도의 경우
    #         # 주식 잔고가 있는지 확인하여 없으면 False return
    #         if self.num_stocks <= 0:
    #             validity = False
    #     return validity
    #
    # def decide_trading_unit(self, action, curr_price):   # 매수 단위 결정
    #     if action == Agent.ACTION_BUY:    # 매수의 경우
    #         trading_unit = int(self.balance / (curr_price * (1 + self.TRADING_CHARGE)))   # 가능한 최대 주식 매수
    #     elif action == Agent.ACTION_SELL:  # 매도의 경우
    #         trading_unit = self.num_stocks  # 보유수량 전부 매도
    #     else:
    #         trading_unit = 0               # Hold
    #     return trading_unit

    # def decide_trading_unit(self, confidence):   # 확률이 높을수록 더 많은 주식 거래  --> 차라리 주식 변동 예상 폭이 클때로 변경이 좋을 것임(다음 version 고려)
    #     if np.isnan(confidence):
    #         return self.min_trading_unit
    #     added_traiding = max(min(
    #         int(confidence * (self.max_trading_unit - self.min_trading_unit)),
    #         self.max_trading_unit-self.min_trading_unit
    #     ), 0)
    #     return self.min_trading_unit + added_traiding

    #def act(self, action, confidence):
    # def act(self, action):      # Agent 가 결정한 행동을 수행
    #     if not self.validate_action(action):     # action 을 할 수 없는 경우 관망
    #         action = Agent.ACTION_HOLD
    #
    #     # 환경에서 현재 가격 얻기 (매일의 종가로 매매를 가정함)
    #     curr_price = self.environment.get_price()
    #
    #     # 즉시 보상 초기화
    #     self.immediate_reward = 0
    #
    #     # 매수/매도시 거래할 주식 수 결정
    #     trading_unit = self.decide_trading_unit(action, curr_price)
    #
    #     # 매수
    #     if action == Agent.ACTION_BUY:
    #         balance = self.balance - curr_price * (1 + self.TRADING_CHARGE) * trading_unit
    #         # 보유 현금이 모자라거나 최대 trading_unit 을 초과한 경우 가능한 만큼 최대한 매수
    #         # if balance < 0:
    #         #     trading_unit = max(min(
    #         #         int(self.balance / (
    #         #             curr_price * (1 + self.TRADING_CHARGE))), self.max_trading_unit),
    #         #         self.min_trading_unit
    #         #     )
    #         # 수수료를 적용하여 총 매수 금액 산정
    #         buy_amount = curr_price * (1 + self.TRADING_CHARGE) * trading_unit
    #         self.balance -= buy_amount       # 보유 현금을 갱신
    #         self.num_stocks += trading_unit  # 보유 주식 수를 갱신
    #         self.num_buy += 1                # 매수 횟수 증가
    #         self.buy_charge += curr_price * self.TRADING_CHARGE * trading_unit   # 매수 수수료 누적
    #     # 매도
    #     elif action == Agent.ACTION_SELL:
    #         # 보유 주식이 모자랄 경우 가능한 만큼 최대한 매도
    #         #trading_unit = min(trading_unit, self.num_stocks)
    #         sell_amount = curr_price * (
    #             1 - (self.TRADING_TAX + self.TRADING_CHARGE)) * trading_unit
    #         self.num_stocks -= trading_unit  # 보유 주식 수를 갱신
    #         self.balance += sell_amount      # 보유 현금을 갱신
    #         self.num_sell += 1               # 매도 횟수 증가
    #         self.sell_charge += curr_price * self.TRADING_CHARGE * trading_unit   # 매도 수수료 누적
    #         self.sell_tax += curr_price * self.TRADING_TAX * trading_unit         # 거래세 누적
    #     # 홀딩
    #     elif action == Agent.ACTION_HOLD:
    #         self.num_hold += 1  # 홀딩 횟수 증가
    #     else:
    #         print("invalid action", action)
    #         quit()
    #
    #     # 포트폴리오 가치 갱신
    #     self.portfolio_value = self.balance + curr_price * self.num_stocks
    #     # 주식 보유 비율 계산
    #     num_stocks_ratio = self.num_stocks / (self.balance / curr_price + self.num_stocks)
    #     #-------- 초기 포트폴리오 가치와 현재 포트폴리오 가치 비교하여 손익 비율 계산
    #     profitloss = (self.portfolio_value - self.base_portfolio_value) / self.base_portfolio_value
    #
    #     # 즉시 보상 판단
    #     self.immediate_reward = 1 if profitloss >= 0 else -1
    #
    #     # 지연 보상 판단
    #     if profitloss > self.delayed_reward_threshold:
    #         delayed_reward = 1
    #         # 목표 수익률 달성하여 기준 포트폴리오 가치 갱신
    #         self.base_portfolio_value = self.portfolio_value
    #     elif profitloss < -self.delayed_reward_threshold:
    #         delayed_reward = -1
    #         # 손실 기준치를 초과하여 기준 포트폴리오 가치 갱신
    #         self.base_portfolio_value = self.portfolio_value
    #     else:
    #         delayed_reward = 0
    #
    #     self.base_stock_price = curr_price   # 기준 포트폴리오 가치 갱신 시점의 주가 save
    #
    #     return self.immediate_reward, delayed_reward
