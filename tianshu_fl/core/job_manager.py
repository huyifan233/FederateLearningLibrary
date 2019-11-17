import queue
from tianshu_fl.entity.runtime_config import RuntimeConfig
import threading


lock = threading.RLock()

class JobManager(object):

    def __init__(self):
        if RuntimeConfig.WAIT_JOB_LIST is None:
            RuntimeConfig.WAIT_JOB_LIST = []
        if RuntimeConfig.EXEC_JOB_QUEUE is None:
            RuntimeConfig.EXEC_JOB_QUEUE = queue.Queue()
        if RuntimeConfig.PENDING_JOB_QUEUE is None:
            RuntimeConfig.PENDING_JOB_QUEUE = queue.Queue()

    def submit_job(self, job):

        with lock:
            RuntimeConfig.WAIT_JOB_LIST.append(job)

    def prepare_job(self, job):
        with lock:
            RuntimeConfig.WAIT_JOB_LIST.remove(job)
            RuntimeConfig.PENDING_JOB_QUEUE.put(job)

    def exec_job(self):
        with lock:
            job = RuntimeConfig.PENDING_JOB_QUEUE.get()
            RuntimeConfig.EXEC_JOB_QUEUE.put(job)

    def complete(self):
        with lock:
            RuntimeConfig.EXEC_JOB_QUEUE.get()