import os
import locale
import logging

logger = logging.getLogger(__name__)
locale.setlocale(locale.LC_ALL, 'ko_KR.UTF-8')
print(locale.currency(100, grouping=True))
print("[KRW :" + locale.currency(100, grouping=True))
file_handler = logging.FileHandler(filename=os.path.join(
        "logs", "test.log"), encoding='utf-8')
stream_handler = logging.StreamHandler()
file_handler.setLevel(logging.DEBUG)
stream_handler.setLevel(logging.INFO)
logging.basicConfig(format="%(message)s",
                    handlers=[file_handler, stream_handler], level=logging.DEBUG)
logger.info("[KRW :".format(locale.currency(100, grouping=True)))
            #
            # logger.info("[Epoch %s/%s]\tEpsilon:%.4f\t#Expl.:%d/%d\t"
            #             "#Buy:%d\t#Sell:%d\t#Hold:%d\t"
            #             "#Stocks:%d\tPV:%s\t"
            #             "POS:%s\tNEG:%s\tLoss:%10.6f" % (
            #                 epoch_str, num_epoches, epsilon, exploration_cnt, itr_cnt,
            #                 self.agent.num_buy, self.agent.num_sell, self.agent.num_hold,
            #                 self.agent.num_stocks,
            #                 locale.currency(self.agent.portfolio_value, grouping=True),
            #                 pos_learning_cnt, neg_learning_cnt, loss))
