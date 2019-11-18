
import threading



class TianshuFlServer(threading.Thread):

    def __init__(self):
       pass



class TianshuFlStandaloneServer(TianshuFlServer):
    def __init__(self):
        super(TianshuFlStandaloneServer, self).__init__()

    def run(self):
        pass





class TianshuFlClusterServer(TianshuFlServer):

    def __init__(self):
        super(TianshuFlClusterServer, self).__init__()

    def run(self):
        pass








