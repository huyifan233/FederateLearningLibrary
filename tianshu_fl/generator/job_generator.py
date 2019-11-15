
import torch.nn.functional as F


class Job(object):
    def __init__(self):
        self.server_host = None
        self.job_id = None




    def set_server_host(self, server_host):
        self.server_host = server_host

    def set_job_id(self, job_id):
        self.job_id = job_id

