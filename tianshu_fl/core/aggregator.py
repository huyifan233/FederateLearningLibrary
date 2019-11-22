
import os
import pickle
import torch
import time
from concurrent.futures import ThreadPoolExecutor
from tianshu_fl.core.strategy import WorkModeStrategy
from tianshu_fl.generator.job_generator import Job

LOCAL_AGGREGATE_FILE = "tmp_aggregate_pars\\avg_pars"

class Aggregator(object):
    def __init__(self, work_mode, job_path, base_model_path, concurrent_num=3):
        self.job_path = job_path
        self.base_model_path = base_model_path
        self.aggregate_executor_pool = ThreadPoolExecutor(concurrent_num)
        self.work_mode = work_mode

    def load_job_list(self, job_path):
        job_list = []
        for job_file in os.listdir(job_path):
            f = open(job_path+"\\"+job_file, "rb")
            job = pickle.load(f)
            job_list.append(job)
        return job_list

    def load_aggregate_model_pars(self, job_model_pars_path):
        job_model_pars = []
        one_model_par_files_len = 0
        #print("job_model_pars_path: ", job_model_pars_path)
        for f in os.listdir(job_model_pars_path):
            if f.find("models_") != -1:
                one_model_par_path = job_model_pars_path + "\\"+f
                #print("one_model_par_path: ", one_model_par_path)
                one_model_par_files = os.listdir(one_model_par_path)
                one_model_par_files_len = len(one_model_par_files)
                if one_model_par_files and len(one_model_par_files) != 0:
                    model_par = torch.load(one_model_par_path+"\\"+os.listdir(one_model_par_path)[-1])
                    job_model_pars.append(model_par)
                else:
                    # wait for other clients finish training
                    return None, 0

        return job_model_pars, one_model_par_files_len



class FedAvgAggregator(Aggregator):
    def __init__(self, work_mode, job_path, base_model_path):
        super(FedAvgAggregator, self).__init__(work_mode, job_path, base_model_path)
        self.fed_step = 0
    def aggregate(self):
        print(self.work_mode)
        if self.work_mode == WorkModeStrategy.WORKMODE_STANDALONE:
            while True:
                job_list = self.load_job_list(self.job_path)
                for job in job_list:
                    job_model_pars, fed_step = self.load_aggregate_model_pars(self.base_model_path + "\\models_{}".format(job.get_job_id()))
                    if self.fed_step != fed_step:
                        futures = self.aggregate_executor_pool.submit(self.exec, self.work_mode, job_model_pars, self.base_model_path, job.get_job_id(), fed_step)
                        print(futures.result())
                        self.fed_step = fed_step
                time.sleep(5)
        else:
            pass


    def exec(self, work_mode, job_model_pars, base_model_path, job_id, fed_step):
        avg_model_par = job_model_pars[0]
        for key in avg_model_par.keys():
            for i in range(1, len(job_model_pars)):
                avg_model_par[key] += job_model_pars[i][key]
            avg_model_par[key] = torch.div(avg_model_par[key], len(job_model_pars))
        if work_mode == WorkModeStrategy.WORKMODE_STANDALONE:
            tmp_aggregate_dir = base_model_path + "\\models_{}".format(job_id)
            tmp_aggregate_path = base_model_path +"\\models_{}\\{}_{}".format(job_id, LOCAL_AGGREGATE_FILE, fed_step)
            if not os.path.exists(tmp_aggregate_dir):
                os.makedirs(tmp_aggregate_path)
            torch.save(avg_model_par, tmp_aggregate_path)
        else:
            pass
        print("aggregate success!!")

class DistillationAggregator(Aggregator):
    def __init__(self):
        super(DistillationAggregator, self).__init__()



    def aggregate(self):
        pass