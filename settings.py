import time
import datetime
import locale
import logging
import os
import platform

import pandas as pd

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Settings for Project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# WINDOWS = [5, 10, 20, 60, 120]
WINDOWS = [5, 10, 20]

# '005380'-현대차, '005930'-삼성전자, '051910'-LG화학, '035420'-NAVER, '030200'-KT, '000660'-SK하이닉스
STOCK_CODE = '035420'
MARKET_CODE = '001'    # '001'-KOSPI

TRAINING_START_DATE   = '2001-01-01'
TRAINING_END_DATE     = '2016-12-31'
SIMULATION_START_DATE = '2017-01-01'
SIMULATION_END_DATE   = '2017-12-31'

LEARNING = True
SIMULATION = True

# # 매매 수수료 및 세금
# TRADING_CHARGE = 0.00015  # 거래 수수료(일반적으로 0.015%)
# TRADING_TAX = 0.003  # 거래세 0.3%

# 행동
BUY  = 0  # 매수
SELL = 1  # 매도
HOLD = 2  # 홀딩
ACTIONS = [BUY, SELL, HOLD]
ACTION_SIZE = len(ACTIONS)       # 인공 신경망 출력값의 개수

# INITIAL_BALANCE = int(('10,000,000').replace(',',''))
# MIN_TRADING_UNIT = 1
# MAX_TRADING_UNIT = 1

REWARD_THRESHOLD = 0.05   # 5%  초과면 보상 or penalty

MAX_EPISODES = 1000
BATCH_SIZE   = 10         # 적절한 value ?

REPLAY_MEMORY = 500       # 적절한 value ?
EPISODE_BUFFER_SIZE = 30  # 적절한 value ?
LEARNING_RATE = 0.01

GAMMA = 0.95             # 적절한 value ?
EPSILON = 1.0            #100% 로 시작
EPSILON_DECAY = 0.995

# Date Time Format
timestr = None
FORMAT_DATE = "%Y%m%d"
FORMAT_DATETIME = "%Y%m%d%H%M%S"

# 로케일 설정
if 'Darwin' in platform.system():
    locale.setlocale(locale.LC_ALL, 'ko_KR.UTF-8')
elif 'Windows' in platform.system():
    locale.setlocale(locale.LC_ALL, '')

# Settings on Logging
def get_today_str():
    today = datetime.datetime.combine(datetime.date.today(), datetime.datetime.min.time())
    today_str = today.strftime('%Y%m%d')
    return today_str

def get_time_str():
    global timestr
    timestr = datetime.datetime.fromtimestamp(
        int(time.time())).strftime(FORMAT_DATETIME)
    return timestr

# 로그 기록
log_dir = os.path.join(BASE_DIR, 'logs/%s' % STOCK_CODE)
timestr = get_time_str()
if not os.path.exists('logs/%s' % STOCK_CODE):
    os.makedirs('logs/%s' % STOCK_CODE)
file_handler = logging.FileHandler(filename=os.path.join(
    log_dir, "%s_%s.log" % (STOCK_CODE, timestr)), encoding='utf-8')
stream_handler = logging.StreamHandler()  # send logging output to stdout
file_handler.setLevel(logging.DEBUG)
stream_handler.setLevel(logging.INFO)
logging.basicConfig(format="%(message)s",
                    handlers=[file_handler, stream_handler], level=logging.DEBUG)
