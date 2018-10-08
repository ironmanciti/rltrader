import os
import locale
import logging
from sqlalchemy import create_engine
import pandas as pd

engine = create_engine("mysql+pymysql://yjoh:1234@localhost/stockdb?charset=utf8",convert_unicode=True)

conn = engine.connect()

stock_code = '035420'

chart_data = pd.read_sql("select dailycandle.date, dailycandle.open, dailycandle.high, dailycandle.low,\
            dailycandle.close, dailycandle.volume, credit_ratio, foreigner_net_buy, inst_net_buy\
            from dailycandle, dailyprice where dailycandle.code ='" + stock_code +
            "' and dailycandle.code ='" + stock_code + "' and dailycandle.date = dailyprice.date \
            order by dailycandle.date", con=engine)
conn.close()
