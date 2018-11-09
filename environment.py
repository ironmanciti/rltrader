import numpy as np
from settings import *

class Environment:        # 종가의 위치

    def __init__(self, chart_data=None):     # policy_learner 의 init 에서 생성
        self.chart_data = chart_data
        self.idx        = None
        self.chart      = None
        self.PRICE_IDX  = 4
        self.buy_price  = 0
        self.buy_position  = 0
        self.sell_price = 0
        self.sell_position = 0
        self.reward     = 0
        self.done       = False

    def reset(self):
        self.idx = np.random.randint(len(self.chart_data) / 2)  # 전체 data 1st half 임의 날자에서 시작
        self.chart = self.chart_data.iloc[self.idx].values
        self.buy_price  = 0
        self.buy_position  = 0
        self.sell_price = 0
        self.sell_position = 0
        self.reward = 0
        self.done  = False
        self.state = np.append(self.chart, [self.buy_price, self.buy_position,
                               self.sell_price, self.sell_position])
        return self.state

    def step(self, action):
        if len(self.chart_data) > self.idx + 1:
            self.idx += 1                            # 다음날로 이동
            self.chart = self.chart_data.iloc[self.idx].values
        else:
            self.reward = 0
            self.done   = True
            return  self.state, self.reward, self.done

        if action == SELL:

            if self.buy_position == 1:   # 이미 매수 상태인 경우는 단순 매도

                if self.chart[self.PRICE_IDX] > self.buy_price * (1 + REWARD_THRESHOLD):
                    self.buy_price  = 0
                    self.buy_position  = 0
                    self.reward = 1
                    self.done   = True
                else:
                    self.buy_price  = 0
                    self.buy_position  = 0
                    self.reward = -1
                    self.done   = True
            else:                        # 매수 상태가 아닌 경우
                if self.sell_position == 0:  # 공매도 상태가 아닌 경우 공매도
                    self.sell_price = self.chart[self.PRICE_IDX]
                    self.sell_position = 1
                    self.reward = 0
                    self.done   = False
                else:                       # 이미 공매도 상태인 경우 pass
                    self.reward = 0
                    self.done   = False

        elif action == BUY:

            if self.sell_position == 1:   # 이미 매도 상태인 경우는 단순 매수

                if self.chart[self.PRICE_IDX] < self.sell_price * (1 - REWARD_THRESHOLD):
                    self.sell_price = 0
                    self.sell_position = 0
                    self.reward = 1
                    self.done   = True
                else:
                    self.sell_price = 0
                    self.sell_position = 0
                    self.reward = -1
                    self.done   = True
            else:                        # 매도 상태가 아닌 경우
                if self.buy_position == 0:  # 매수 상태가 아닌 경우 매수
                    self.buy_price  = self.chart[self.PRICE_IDX]
                    self.buy_position = 1
                    self.reward = 0
                    self.done   = False
                else:                       # 이미 매수 상태인 경우 pass
                    self.reward = 0
                    self.done   = False

        # elif action == HOLD:
        #     self.reward = 0
        #     self.done   = False
        else:
            print("Invalid action : ", action)

        if self.buy_position == 1 and self.sell_position == 1:
            self.reward = 0

        self.state = np.append(self.chart, [self.buy_price, self.buy_position,
                          self.sell_price, self.sell_position])

        return self.state, self.reward, self.done


    # def get_price(self):
    #     if self.observation is not None:
    #         return self.observation[self.PRICE_IDX]  # 종가 return
    #     return None

    # def set_chart_data(self, chart_data):
    #     self.chart_data = chart_data
