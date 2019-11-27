import torch
import threading
import pickle
import os
import inspect
from tianshu_fl.entity import runtime_config
from tianshu_fl.entity.job import Job
from tianshu_fl.utils.utils import JobUtils
from tianshu_fl.core.strategy import WorkModeStrategy

lock = threading.RLock()

class JobManager(object):

    def __init__(self, job_path):
       self.job_path = job_path


    def generate_job(self, work_mode, train_code_strategy, model, iter):
        with lock:
            job = Job()
            job.set_job_id(JobUtils.generate_job_id())
            if work_mode == WorkModeStrategy.WORKMODE_STANDALONE:
                job.set_server_host("localhost:8080")
            else:
                job.set_server_host("")
            job.set_train_strategy(train_code_strategy)
            job.set_train_model(inspect.getsourcefile(model))
            job.set_train_model_class_name(model.__name__)
            job.set_iterations(iter)
            return job

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

