import torch
import threading
import pickle
import os
from tianshu_fl.entity import runtime_config


lock = threading.RLock()

class JobManager(object):

    def __init__(self, job_path):
       self.job_path = job_path

    def submit_job(self, job, model, model_path):

        with lock:
            # create model dir of this job
            job_model_dir = model_path + "\\"+"models_{}".format(job.get_job_id())
            if not os.path.exists(job_model_dir):
                os.makedirs(job_model_dir)
            torch.save(model, job_model_dir+"\\"+"{}.pt".format(job.get_job_id()))
            f = open(self.job_path+"\\"+"job_{}".format(job.get_job_id()), "wb")
            pickle.dump(job, f)
            print("job {} added successfully".format(job.get_job_id()))


    def prepare_job(self, job):
        with lock:
            runtime_config.remove_waiting_job(job)
            runtime_config.add_pending_job(job)

    def exec_job(self, job):
        with lock:
            exec_job = runtime_config.remove_pending_job(job)
            runtime_config.add_exec_job(job)

    def complete(self):
        with lock:
            runtime_config.get_exec_job()

