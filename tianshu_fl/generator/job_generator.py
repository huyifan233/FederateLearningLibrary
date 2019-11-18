


class Job(object):

    def __init__(self):
        self.server_host = None
        self.job_id = None
        self.train_strategy = None
        self.train_model = None

    def set_server_host(self, server_host):
        self.server_host = server_host

    def set_job_id(self, job_id):
        self.job_id = job_id

    def get_job_id(self):
        return self.job_id

    def set_train_strategy(self, train_strategy):
        self.train_strategy = train_strategy

    def set_train_model(self, train_model):
        self.train_model = train_model


