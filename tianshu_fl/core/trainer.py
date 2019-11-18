
import threading
from tianshu_fl.core.strategy import WorkModeStrategy
from tianshu_fl.entity.runtime_config import RuntimeConfig



class TrainerObserver(object):
    def __init__(self, work_mode):
        self.work_mode = work_mode



    def start(self):
        if self.work_mode == WorkModeStrategy.WORKMODE_STANDALONE:
            pending_job_list = RuntimeConfig.PENDING_JOB_LIST




class Trainer(threading.Thread):
    def __init__(self):
        super(Trainer, self).__init__()