


class Job(object):

    def __init__(self):
        self.server_host = None
        self.job_id = None
        self.train_strategy = None
        self.train_model = None
        self.train_model_class_name = None
        self.iterations = None
        self.status = None

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

    def set_train_model_class_name(self, train_model_class_name):
        self.train_model_class_name = train_model_class_name

    def get_server_host(self):
        return self.server_host

    def get_train_strategy(self):
        return self.train_strategy

    def get_train_model(self):
        return self.train_model

    def set_iterations(self, iterations):
        self.iterations = iterations

    def get_iterations(self):
        return self.iterations