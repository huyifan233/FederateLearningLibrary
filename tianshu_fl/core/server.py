
import threading
from tianshu_fl.entity.runtime_config import RuntimeConfig

class TianshuFlServer(threading.Thread):

    def __init__(self):
       pass



class TianshuFlStandaloneServer(TianshuFlServer):
    def __init__(self):
        super(TianshuFlStandaloneServer, self).__init__()



    def run(self):
        queue = RuntimeConfig.WAIT_JOB_QUEUE
        while True:
            if not queue.empty():
                pass




class TianshuFlClusterServer(TianshuFlServer):

    def __init__(self):
        super(TianshuFlClusterServer, self).__init__()

    def run(self):
        pass








