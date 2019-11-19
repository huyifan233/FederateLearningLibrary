import threading
import time
import os
from tianshu_fl.entity import runtime_config
from tianshu_fl.generator.utils import JobUtils
from tianshu_fl.core.strategy import WorkModeStrategy




class JobDetector(threading.Thread):

    def __init__(self, time=1, work_mode=WorkModeStrategy.WORKMODE_STANDALONE):
        super(JobDetector, self).__init__()
        self.time = time
        self.work_mode = work_mode


    def run(self):
        while True:
            job_list = JobDetector.list_all_jobs("")
            if len(job_list) != 0:
                for job in job_list:
                    if self.work_mode != WorkModeStrategy.WORKMODE_STANDALONE:
                        job = JobUtils.serialize(job)
            time.sleep(self.time)

    @staticmethod
    def list_all_jobs(self, job_path):
        return os.listdir(job_path)
