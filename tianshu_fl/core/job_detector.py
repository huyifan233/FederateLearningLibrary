
from tianshu_fl.entity.runtime_config import RuntimeConfig
from tianshu_fl.generator.utils import JobUtils
from tianshu_fl.core.strategy import WorkModeStrategy
import threading
import time

class JobDetector(threading.Thread):

    def __init__(self, time=1, work_mode=WorkModeStrategy.WORKMODE_STANDALONE):
        super(JobDetector, self).__init__()
        self.time = time
        self.work_mode = work_mode

    def run(self):
        wait_job_list = RuntimeConfig.WAIT_JOB_LIST
        pending_job_list = RuntimeConfig.PENDING_JOB_LIST
        while True:
            if len(wait_job_list) != 0:
                for job in wait_job_list:
                    if self.work_mode == WorkModeStrategy.WORKMODE_STANDALONE:
                        job = JobUtils.serialize(job)
                    pending_job_list.append(job)
                wait_job_list.clear()
            time.sleep(self.time)
