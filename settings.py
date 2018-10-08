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


# # Settings for Templates
# TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
#
#
# # Settings for Static
# STATIC_DIR = os.path.join(BASE_DIR, "static")
# STATIC_URL = "/static/"
#
#
# # Settings for Data
# DATA_DIR = os.path.join(BASE_DIR, "database")


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
