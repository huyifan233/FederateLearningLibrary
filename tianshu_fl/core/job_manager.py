import queue
from tianshu_fl.entity.runtime_config import RuntimeConfig

class JobManager(object):

    def __init__(self):
        if RuntimeConfig.WAIT_JOB_QUEUE is None:
            RuntimeConfig.JOB_QUEUE = queue.Queue()
        if RuntimeConfig.EXEC_JOB_QUEUE is None:
            RuntimeConfig.EXEC_JOB_QUEUE = queue.Queue()


    def submit_job(self, job):

        RuntimeConfig.WAIT_JOB_QUEUE.put(job)

