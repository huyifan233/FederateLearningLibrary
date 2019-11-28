import torch
import threading
import pickle
import os, json
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
            #server_host, job_id, train_strategy, train_model, train_model_class_name, iterations
            job = Job(None, JobUtils.generate_job_id(), train_code_strategy, inspect.getsourcefile(model),
                      model.__name__, iter)
            if work_mode == WorkModeStrategy.WORKMODE_STANDALONE:
                job.set_server_host("localhost:8080")
            else:
                job.set_server_host("")

            return job

    def submit_job(self, job, model, model_path):

        with lock:
            # create model dir of this job
            job_model_dir = model_path + "\\"+"models_{}".format(job.get_job_id())
            if not os.path.exists(job_model_dir):
                os.makedirs(job_model_dir)
            torch.save(model.state_dict(), job_model_dir+"\\"+"init_model_pars_{}".format(job.get_job_id()))
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

    @staticmethod
    def get_job_list(job_path):
        job_list = []
        for job_file in os.listdir(job_path):
            job_file_path = job_path + "\\" + job_file
            with open(job_file_path, "rb") as f:
                job = pickle.load(f)
                job_list.append(job)
        return job_list