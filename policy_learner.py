import os
import locale
import logging
import numpy as np
import settings
from environment import Environment
from agent import Agent
from policy_network import PolicyNetwork
from visualizer import Visualizer


logger = logging.getLogger(__name__)
locale.setlocale(locale.LC_ALL, 'ko_KR.UTF-8')


class PolicyLearner:

    def __init__(self, stock_code, chart_data, training_data=None,
                 min_trading_unit=0, max_trading_unit=0,
                 delayed_reward_threshold=0, lr=0):
        self.stock_code = stock_code  # 종목코드
        self.chart_data = chart_data
        self.environment = Environment(chart_data)  # 환경 객체 (차트데이터를 순서대로 읽으면서 주가, 거래량 제공)
        # 에이전트 객체
        self.agent = Agent(self.environment,
                           min_trading_unit=min_trading_unit,
                           max_trading_unit=max_trading_unit,
                           delayed_reward_threshold=delayed_reward_threshold)
        self.training_data = training_data  # 학습 데이터
        self.sample = None
        self.training_data_idx = -1
        # 정책 신경망; 입력 크기 = 학습 데이터의 크기 + 에이전트 상태 크기
        self.num_features = self.training_data.shape[1] + self.agent.STATE_DIM
        self.policy_network = PolicyNetwork(
            input_dim=self.num_features, output_dim=self.agent.NUM_ACTIONS, lr=lr)
        self.visualizer = Visualizer()  # 가시화 모듈

    def reset(self):       # epoch 마다 호출
        self.sample = None
        self.training_data_idx = -1

    def fit(self, balance=0, num_epoches=0, max_memory=0,
        discount_factor=0, start_epsilon=0, learning=0):   # 모델훈련: learning=True, trade: False
        logger.info("LR: {lr}, DF: {discount_factor}, "
                    "TU: [{min_trading_unit}, {max_trading_unit}], "
                    "DRT: {delayed_reward_threshold}".format(
            lr=self.policy_network.lr,
            discount_factor=discount_factor,
            min_trading_unit=self.agent.min_trading_unit,
            max_trading_unit=self.agent.max_trading_unit,
            delayed_reward_threshold=self.agent.delayed_reward_threshold
        ))

        # 가시화 준비
        # 차트 데이터는 변하지 않으므로 미리 가시화
        self.visualizer.prepare(self.environment.chart_data)   # 일봉차트

        # 가시화 결과 저장할 폴더 준비
        epoch_summary_dir = os.path.join(
            settings.BASE_DIR, 'epoch_summary/%s/epoch_summary_%s' % (
                self.stock_code, settings.timestr))       # local time 을 "%Y%m%d%H%M%S" 형식으로 표시
        if not os.path.isdir(epoch_summary_dir):
            os.makedirs(epoch_summary_dir)

        # 에이전트 초기 자본금 설정
        self.agent.set_balance(balance)

        # 학습에 대한 정보 초기화
        max_portfolio_value = 0
        epoch_win_cnt = 0

        # 학습 반복
        for epoch in range(num_epoches):
            # 에포크 관련 정보 초기화
            loss = 0.
            itr_cnt = 0              # epoch 내에서 수행한 반복수 (sample data 수)
            win_cnt = 0              # 수익발생 횟수
            exploration_cnt = 0      # 무작위 투자 횟수
            batch_size = 0
            pos_learning_cnt = 0     # 긍정적 지연보상 횟수
            neg_learning_cnt = 0     # 부정적 지연보상 횟수

            # 메모리 초기화
            memory_sample = []
            memory_action = []
            memory_reward = []
            memory_prob = []
            memory_pv = []
            memory_num_stocks = []
            memory_exp_idx = []
            memory_learning_idx = []

            # 환경, 에이전트, 정책 신경망 초기화
            self.environment.reset()
            self.agent.reset()
            self.policy_network.reset()
            self.reset()

            # 가시화 초기화
            self.visualizer.clear([0, len(self.chart_data)])     # figure 초기화 & x, y 축 설정

            # 학습을 진행할 수록 탐험 비율 감소
            if learning:
                epsilon = start_epsilon * (1. - float(epoch) / (num_epoches - 1))
            else:
                epsilon = 0

            while True:
                # 샘플 생성
                next_sample = self._build_sample()
                if next_sample is None:
                    break

                # 정책 신경망 또는 탐험(무작위투자)에 의한 행동 결정
                #action, confidence, exploration = self.agent.decide_action(
                action, exploration = self.agent.decide_action(
                    self.policy_network, self.sample, epsilon)

                # 결정한 행동을 수행하고 즉시 보상과 지연 보상 획득. 손익> 0 면 1 / 손익 < 0 면 -1 / 손익=0 면 0 반환
                #immediate_reward, delayed_reward = self.agent.act(action, confidence)
                immediate_reward, delayed_reward = self.agent.act(action)

                # 행동 및 행동에 대한 결과를 기억하여 training data 를 만들 data
                memory_sample.append(next_sample)
                memory_action.append(action)
                memory_reward.append(immediate_reward)
                memory_pv.append(self.agent.portfolio_value)
                memory_num_stocks.append(self.agent.num_stocks)
                memory = [(
                    memory_sample[i],
                    memory_action[i],
                    memory_reward[i])
                    for i in list(range(len(memory_action)))[-max_memory:]    # max_memory 만큼만 training data 로 사용
                ]
                if exploration:        # 무작위 투자인 경우
                    memory_exp_idx.append(itr_cnt)
                    memory_prob.append([np.nan] * Agent.NUM_ACTIONS)
                else:
                    memory_prob.append(self.policy_network.prob)   # agent.decide_action 에서 부른 정책신경망에 의해 결정된 softmax 확률

                # 반복에 대한 정보 갱신
                batch_size += 1
                itr_cnt += 1
                exploration_cnt += 1 if exploration else 0
                win_cnt += 1 if delayed_reward > 0 else 0

                # 학습 모드이고 지연 보상이 존재할 경우 정책 신경망 갱신
                if delayed_reward == 0 and batch_size >= max_memory:      # max_memory 가 다찬경우 즉시보상으로 지연보상을 대체
                    delayed_reward = immediate_reward
                    self.agent.base_portfolio_value = self.agent.portfolio_value
                if learning and delayed_reward != 0:              # 지연보상 발생
                    # 배치 학습 데이터 크기
                    batch_size = min(batch_size, max_memory)       # training data 는 max_memory 크기만큼만 사용
                    # 배치 학습 데이터 생성
                    x, y = self._get_batch(
                        memory, batch_size, discount_factor, delayed_reward)
                    if len(x) > 0:
                        if delayed_reward > 0:       # 긍정적 학습 횟수
                            pos_learning_cnt += 1
                        else:
                            neg_learning_cnt += 1     # 부정적 학습 횟수
                        # -------- 정책 신경망 갱신 (학습 수행) -----------
                        loss += self.policy_network.train_on_batch(x, y)

                        memory_learning_idx.append([itr_cnt, delayed_reward])    # 학습이 진행된 index 저장
                    batch_size = 0

            # 에포크 관련 정보 가시화 (epoch 수)
            num_epoches_digit = len(str(num_epoches))
            epoch_str = str(epoch + 1).rjust(num_epoches_digit, '0')

            self.visualizer.plot(
                epoch_str=epoch_str, num_epoches=num_epoches, epsilon=epsilon,
                action_list=Agent.ACTIONS, actions=memory_action,
                num_stocks=memory_num_stocks, outvals=memory_prob,
                exps=memory_exp_idx, learning=memory_learning_idx,
                initial_balance=self.agent.initial_balance, pvs=memory_pv
            )
            self.visualizer.save(os.path.join(
                epoch_summary_dir, 'epoch_summary_%s_%s.png' % (
                    settings.timestr, epoch_str)))

            # 에포크 관련 정보 로그 기록
            if pos_learning_cnt + neg_learning_cnt > 0:
                loss /= pos_learning_cnt + neg_learning_cnt
            logger.info("[Epoch {}/{}] Epsilon:{:.4f} #Expl.:{:.0f}/{:.0f}  "
                        "#Buy:{:.0f} #Sell:{:.0f} #Hold:{:.0f}  "
                        "#Stocks:{:.0f} PV:KRW{:.0f} "
                        "POS:{} NEG:{} Loss:{:10.6f} BuyCharge:{:.0f} "
                        "SellCharge:{:.0f} Tax:{:.0f}".format(
                            epoch_str, num_epoches, epsilon, exploration_cnt, itr_cnt,
                            self.agent.num_buy, self.agent.num_sell, self.agent.num_hold,
                            self.agent.num_stocks,self.agent.portfolio_value, pos_learning_cnt,
                            neg_learning_cnt, loss, self.agent.buy_charge, self.agent.sell_charge,
                            self.agent.sell_tax))

            # 학습 관련 통계 정보 갱신
            max_portfolio_value = max(
                max_portfolio_value, self.agent.portfolio_value)        # 달성한 최대 PV
            if self.agent.portfolio_value > self.agent.initial_balance:
                epoch_win_cnt += 1

        # 학습 관련 정보 로그 기록
        logger.info("Max PV: KRW %d, \t # Win: %d" % (max_portfolio_value, epoch_win_cnt))

    # training data 생성 (batch size 는 지연보상시마다 매번 다름)
    def _get_batch(self, memory, batch_size, discount_factor, delayed_reward):
        x = np.zeros((batch_size, 1, self.num_features))
        #y = np.full((batch_size, self.agent.NUM_ACTIONS), 0.5)
        y = np.zeros((batch_size, self.agent.NUM_ACTIONS))

        for i, (sample, action, reward) in enumerate(
                reversed(memory[-batch_size:])):
            x[i] = np.array(sample).reshape((-1, 1, self.num_features))      # input data
            if action == self.agent.ACTION_BUY:
                if delayed_reward > 0:
                    y[i, self.agent.ACTION_BUY]  = 1
                    y[i, self.agent.ACTION_SELL] = 0
                    y[i, self.agent.ACTION_HOLD] = 0
                else:
                    y[i, self.agent.ACTION_BUY]  = 0
                    y[i, self.agent.ACTION_SELL] = 1
                    y[i, self.agent.ACTION_HOLD] = 0
            elif action == self.agent.ACTION_SELL:
                if delayed_reward > 0:
                    y[i, self.agent.ACTION_BUY]  = 0
                    y[i, self.agent.ACTION_SELL] = 1
                    y[i, self.agent.ACTION_HOLD] = 0
                else:
                    y[i, self.agent.ACTION_BUY]  = 1
                    y[i, self.agent.ACTION_SELL] = 0
                    y[i, self.agent.ACTION_HOLD] = 0
            elif action == self.agent.ACTION_HOLD:
                if delayed_reward > 0:
                    y[i, self.agent.ACTION_BUY]  = 0
                    y[i, self.agent.ACTION_SELL] = 0
                    y[i, self.agent.ACTION_HOLD] = 1
                else:
                    y[i, self.agent.ACTION_BUY]  = 0.5
                    y[i, self.agent.ACTION_SELL] = 0.5
                    y[i, self.agent.ACTION_HOLD] = 0
            else:
                print("invalid action =", action)
                quit()

            # if discount_factor > 0:
            #     y[i, action] *= discount_factor ** i
        return x, y

    def _build_sample(self):            # 학습데이터를 구성하는 sample 하나 생성
        self.environment.observe()      # next data
        if len(self.training_data) > self.training_data_idx + 1:
            self.training_data_idx += 1
            self.sample = self.training_data.iloc[self.training_data_idx].tolist()
            self.sample.extend(self.agent.get_states())     # 주식보유비율, 포트폴리오 가치 비율을 sample list 에 추가
            return self.sample
        return None

    # 주식투자 simulation
    def trade(self, model_path=None, balance=0, num_epoches=0,max_memory=0,
                       discount_factor=0, start_epsilon=0):
        if model_path is None:
            return
        self.policy_network.load_model(model_path=model_path)      # 학습된 정책 신경망 model load
        self.fit(balance=balance, num_epoches=1, max_memory=max_memory,
                   discount_factor=discount_factor, start_epsilon= start_epsilon, learning=False)
