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


class Utils(object):
    def __init__(self):
        pass


jobCount = JobIdCount(init_value=0)

class JobUtils(Utils):
    def __init__(self):
        super(JobUtils, self).__init__()

    @staticmethod
    def generate_job_id(self):
        return '{}{}'.format(datetime.datetime.now().strftime("%Y%m%d%H%M%S%f"), jobCount.incr())



