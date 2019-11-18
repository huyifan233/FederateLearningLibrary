
from tianshu_fl.core.server import TianshuFlStandaloneServer
from tianshu_fl.core.job_detector import JobDetector
from tianshu_fl.core.strategy import WorkModeStrategy
if __name__ == "__main__":

    JobDetector(1, WorkModeStrategy.WORKMODE_STANDALONE).start()
    TianshuFlStandaloneServer().start()

