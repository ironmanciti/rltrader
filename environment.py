import numpy as np
from settings import *

class Environment:

<<<<<<< HEAD
    def __init__(self, chart_data=None):     # policy_learner 의 init 에서 생성
        self.chart_data = chart_data
        self.observation = None             # 현재 관측치
        self.idx = -1                       # chart data 에서의 현재 위치

    def reset(self):
        self.observation = None             # 초기화 --> chart data 의 처음으로
        self.idx = -1

    def observe(self):
        if len(self.chart_data) > self.idx + 1:
            self.idx += 1                   # 하루 앞으로 이동
            self.observation = self.chart_data.iloc[self.idx]
            return self.observation
        return None

    def get_price(self):
        if self.observation is not None:
            return self.observation[self.PRICE_IDX]  # 종가 return
        return None
=======
    def __init__(self, chart_data, scaled_data):
        self.chart_data = chart_data
        self.scaled_data = scaled_data
        self.idx        = -1
        self.row      = None
        self.chart    = None
        self.DATE_IDX  = 0      # 날자의 위치
        self.PRICE_IDX  = 4     # 종가의 위치
        self.buy_price  = 0
        self.long_position  = False
        self.sell_price = 0
        self.short_position = False
        self.buy_info  = {"price": 0, "date": None}
        self.sell_info = {"price": 0, "date": None}
        self.reward     = 0
        self.done       = False
        self.eof        = False
        self.state      = None

    def reset(self, run_mode):
        if run_mode == "LEARNING":
            self.idx = np.random.randint(len(self.scaled_data)) - EPISODE_BUFFER_SIZE
        elif run_mode == "SIMULATION":
            self.idx = +1
        else:
            return
        self.row = self.scaled_data.iloc[self.idx].values
        self.chart = self.chart_data.iloc[self.idx].values
        self.buy_price  = 0
        self.long_position  = False
        self.sell_price = 0
        self.short_position = False
        self.buy_info  = {"price": 0, "date": None}
        self.sell_info = {"price": 0, "date": None}
        self.reward = 0
        self.done  = False
        self.profitloss = 0

        state_buffer = []
        for i in range(1, EPISODE_BUFFER_SIZE):
            self.idx += 1
            if self.idx >= len(self.scaled_data):
                pass
            else:
                chart = self.scaled_data.iloc[self.idx].values
                state = np.append(chart, [self.buy_price, self.long_position,\
                                  self.sell_price, self.short_position])
                state_buffer.append(state)

        if len(state_buffer) > 0:
            self.state = np.array(state_buffer)
>>>>>>> 6ad1fe99b5f6f32b80563112803f402ef76f9fa6

        return self.state

    def step(self, action):

        self.reward = 0
        self.done   = False

        if len(self.scaled_data) > self.idx + 1:
            self.idx += 1                            # 다음날로 이동
            self.row = self.scaled_data.iloc[self.idx].values
            self.chart = self.chart_data.iloc[self.idx].values
        else:
            self.done   = True
            self.eof    = True
            return  self.state, self.reward, self.done, self.profitloss

        if action == BUY:

            if not self.long_position:      # 매수 상태가 아닌 경우 매수
                self.buy_price  = self.chart[self.PRICE_IDX]
                self.long_position = True
                self.buy_info = {"price": self.chart[self.PRICE_IDX], "date": self.chart[self.DATE_IDX]}
                self.reward = 0
                self.done   = False

            if self.short_position:        # 이미 공매도 상태인 경우는 재매수
                str = "공매도 (date {} price {}) / 재매수 (date {} price {})" \
                        .format(self.sell_info['date'], self.sell_info['price'],\
                        self.chart[self.DATE_IDX], self.chart[self.PRICE_IDX])

                self.profitloss = self.sell_price - self.chart[self.PRICE_IDX]

                if self.profitloss >= self.sell_price * REWARD_THRESHOLD:
                    print("win - " + str)
                    self.reward = +10
                elif self.profitloss > 0 and \
                     (self.profitloss < self.sell_price * REWARD_THRESHOLD):
                    print("win - " + str)
                    self.reward = +5
                else:
                    print("lose - " + str)
                    self.reward = -5

                self.done   = True

        if action == SELL:

            if not self.short_position:        # 공매도 상태가 아닌 경우 공매도
                self.sell_price = self.chart[self.PRICE_IDX]
                self.short_position = True
                self.sell_info = {"price": self.chart[self.PRICE_IDX], "date": self.chart[self.DATE_IDX]}
                self.reward = 0
                self.done   = False

            if self.long_position:   # 이미 매수 상태인 경우는 단순 매도
                str = "매수 (date {} price {}) / 매도 (date {} price {})"\
                        .format(self.buy_info['date'], self.buy_info['price'],\
                        self.chart[self.DATE_IDX], self.chart[self.PRICE_IDX])

                self.profitloss = self.chart[self.PRICE_IDX] - self.buy_price

                if self.profitloss >= self.buy_price * REWARD_THRESHOLD:
                    print("win - " + str)
                    self.reward = +10
                elif self.profitloss > 0 and \
                    (self.profitloss < self.buy_price * REWARD_THRESHOLD):
                    self.reward = +5
                    print("win - " + str)
                else:
                    print("lose - " + str)
                    self.reward = -5

                self.done   = True

        if action == HOLD:
            # REWARD_THRESHOLD +, - 이내에서 HOLD 하면 reward +1, 아니면 -5
            if self.long_position:
                if (self.chart[self.PRICE_IDX] < self.buy_price * (1 + REWARD_THRESHOLD)) and \
                   (self.chart[self.PRICE_IDX] > self.buy_price * (1 - REWARD_THRESHOLD)):
                   self.reward = +1
                else:
                   self.reward = -5

            if self.short_position:
                if (self.chart[self.PRICE_IDX] > self.sell_price * (1 - REWARD_THRESHOLD)) and \
                   (self.chart[self.PRICE_IDX] < self.sell_price * (1 + REWARD_THRESHOLD)):
                   self.reward = +1
                else:
                   self.reward = -5

        next_state = np.append(self.row, [self.buy_price, self.long_position,\
                          self.sell_price, self.short_position])
        next_state = next_state.reshape(1, next_state.shape[0])
        self.state = np.vstack([self.state, next_state])

        return self.state, self.reward, self.done, self.profitloss
