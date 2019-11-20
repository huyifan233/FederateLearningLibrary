
from tianshu_fl.core.server import TianshuFlStandaloneServer, TianshuFlClusterServer
from tianshu_fl.core.job_detector import JobDetector
from tianshu_fl.core.strategy import WorkModeStrategy, FedrateStrategy

WORK_MODE = WorkModeStrategy.WORKMODE_STANDALONE
FEDERATE_STRATEGY = FedrateStrategy.FED_AVG


if __name__ == "__main__":

    if WORK_MODE == WorkModeStrategy.WORKMODE_STANDALONE:
        TianshuFlStandaloneServer(FEDERATE_STRATEGY).start()
    else:
        TianshuFlClusterServer(FEDERATE_STRATEGY).start()

