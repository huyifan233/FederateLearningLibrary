
import threading
import time
import os
from tianshu_fl.core.aggregator import FedAvgAggregator
from tianshu_fl.core.strategy import WorkModeStrategy, FedrateStrategy


JOB_PATH = os.path.abspath(".") + "\\res\\jobs"
BASE_MODEL_PATH = os.path.abspath(".") + "\\res\\models"

class TianshuFlServer(threading.Thread):

    def __init__(self):
        super(TianshuFlServer, self).__init__()


class TianshuFlStandaloneServer(TianshuFlServer):
    def __init__(self, federate_strategy):
        super(TianshuFlStandaloneServer, self).__init__()
        if federate_strategy == FedrateStrategy.FED_AVG:
            self.aggregator = FedAvgAggregator(WorkModeStrategy.WORKMODE_STANDALONE, JOB_PATH, BASE_MODEL_PATH)
        else:
           pass


    def run(self):
        self.aggregator.aggregate()





class TianshuFlClusterServer(TianshuFlServer):

    def __init__(self):
        super(TianshuFlClusterServer, self).__init__()

    def run(self):
        pass








