class Environment:

    PRICE_IDX = 4  # 종가의 위치

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

    def set_chart_data(self, chart_data):
        self.chart_data = chart_data
