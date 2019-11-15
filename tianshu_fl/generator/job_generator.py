
import torch.nn.functional as F
import threading
import datetime

class JobIdCount(object):

    lock = threading.RLock()

    def __init__(self, init_value):
        self.value = init_value

    def incr(self, step):
        with JobIdCount.lock:
            self.value += step
            return self.value

jobCount = JobIdCount(init_value=0)

class Job(object):

    def __init__(self):
        self.server_host = None
        self.job_id = None


    def set_server_host(self, server_host):
        self.server_host = server_host

    def set_job_id(self, job_id):
        self.job_id = job_id

    def generate_job_id(self):
        return '{}{}'.format(datetime.datetime.now().strftime("%Y%m%d%H%M%S%f"), jobCount.incr())


