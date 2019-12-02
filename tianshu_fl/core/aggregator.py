
import os
import pickle
import torch
import time
import requests
from concurrent.futures import ThreadPoolExecutor
from tianshu_fl.core.strategy import WorkModeStrategy
from tianshu_fl.core.job_manager import JobManager
from tianshu_fl.entity.runtime_config import WAITING_BROADCAST_AGGREGATED_JOB_LIST, CONNECTED_TRAINER_LIST
LOCAL_AGGREGATE_FILE = "tmp_aggregate_pars\\avg_pars"

class Aggregator(object):
    def __init__(self, work_mode, job_path, base_model_path, concurrent_num=3):
        self.job_path = job_path
        self.base_model_path = base_model_path
        self.aggregate_executor_pool = ThreadPoolExecutor(concurrent_num)
        self.work_mode = work_mode

    # def load_job_list(self, job_path):
    #     job_list = []
    #     for job_file in os.listdir(job_path):
    #         f = open(job_path+"\\"+job_file, "rb")
    #         job = pickle.load(f)
    #         job_list.append(job)
    #     return job_list

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

        while True:
            job_list = JobManager.get_job_list(self.job_path)
            WAITING_BROADCAST_AGGREGATED_JOB_LIST.clear()
            for job in job_list:
                job_model_pars, fed_step = self.load_aggregate_model_pars(self.base_model_path + "\\models_{}".format(job.get_job_id()))
                if self.fed_step != fed_step:
                    futures = self.aggregate_executor_pool.submit(self._exec, job_model_pars, self.base_model_path, job.get_job_id(), fed_step)
                    futures.result()
                    self.fed_step = fed_step
                    WAITING_BROADCAST_AGGREGATED_JOB_LIST.append(job.get_job_id())
                    if job.get_iterations() <= self.fed_step:
                        self._save_final_model_pars(job.get_job_id(), self.base_model_path + "\\models_{}\\tmp_aggregate_pars".format(job.get_job_id()))


            if self.work_mode == WorkModeStrategy.WORKMODE_CLUSTER:
                self._broadcast(WAITING_BROADCAST_AGGREGATED_JOB_LIST, CONNECTED_TRAINER_LIST, self.base_model_path)
            time.sleep(5)



    def _exec(self, job_model_pars, base_model_path, job_id, fed_step):
        avg_model_par = job_model_pars[0]
        for key in avg_model_par.keys():
            for i in range(1, len(job_model_pars)):
                avg_model_par[key] += job_model_pars[i][key]
            avg_model_par[key] = torch.div(avg_model_par[key], len(job_model_pars))
        tmp_aggregate_dir = base_model_path + "\\models_{}".format(job_id)
        tmp_aggregate_path = base_model_path +"\\models_{}\\{}_{}".format(job_id, LOCAL_AGGREGATE_FILE, fed_step)
        if not os.path.exists(tmp_aggregate_dir):
            os.makedirs(tmp_aggregate_path)
        torch.save(avg_model_par, tmp_aggregate_path)

        print("aggregate success!!")


    def _broadcast(self, job_list, connected_client_list, base_model_path):
        aggregated_files, job_ids = self._prepare_upload_aggregate_file(job_list, base_model_path)

        for client in connected_client_list:
            client_url = "http://{}".format(client)
            response = requests.post("/".join([client_url, "aggregatepars"], data=job_ids, files=aggregated_files))
            print(response.json())

    def _prepare_upload_aggregate_file(self, job_list, base_model_path):
        aggregated_files = {}
        job_ids = []
        for job in job_list:
            send_aggregate_filename = "tmp_aggregate_{}".format(job.get_job_id())
            tmp_aggregate_dir = base_model_path + "\\models_{}\\tmp_aggregate_pars".format(job.get_job_id())
            tmp_aggregate_path = tmp_aggregate_dir + "\\" + os.listdir(tmp_aggregate_dir)[-1]
            aggregated_files[send_aggregate_filename] = (send_aggregate_filename, open(tmp_aggregate_path, "rb"))
            job_ids.append(job.get_job_id())
        return aggregated_files, job_ids

    def _save_final_model_pars(self,  job_id, tmp_aggregate_dir):
        job_model_dir = self.base_model_path + "\\models_{}".format(job_id)
        final_model_pars_path = job_model_dir + "\\final_model_pars"
        if not os.path.exists(job_model_dir):
            os.makedirs(job_model_dir)

        with open(final_model_pars_path, "wb") as final_f:
            with open(os.listdir(tmp_aggregate_dir)[-1], "rb") as f:
                for line in f.readlines():
                    final_f.write(line)




class DistillationAggregator(Aggregator):
    def __init__(self):
        super(DistillationAggregator, self).__init__()



    def aggregate(self):
        pass