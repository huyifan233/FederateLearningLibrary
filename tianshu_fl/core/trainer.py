
import time, os, pickle
import importlib
import torch
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor
from tianshu_fl.core.strategy import WorkModeStrategy
from tianshu_fl.entity import runtime_config
from tianshu_fl.core.strategy import RunTimeStrategy


JOB_PATH = "res\\jobs"
LOCAL_MODEL_BASE_PATH = "res\\models\\"
AGGREGATE_PATH = "tmp_aggregate_pars"


class Trainer(object):
    def __init__(self, work_mode, data, client_id, concurrent_num=3):
        self.work_mode = work_mode
        self.data = data
        self.client_id = client_id
        self.concurrent_num = concurrent_num
        self.trainer_executor_pool = ThreadPoolExecutor(self.concurrent_num)
        self.job_path = os.path.abspath(".")+"\\"+JOB_PATH
        self.fed_step = 0
        self.job_iter_dict = {}
        self.is_finish = True


    def start(self):
        if self.work_mode == WorkModeStrategy.WORKMODE_STANDALONE:
            # start train
            while True:
                job_list = self._list_all_jobs(self.job_path, self.job_iter_dict)
                self._prepare_jobs_model(job_list)
                for job in job_list:
                    if self.job_iter_dict.get(job.get_job_id()) is None or self.job_iter_dict.get(job.get_job_id()) != job.get_iterations():
                        self.is_finish = False
                        break
                if self.is_finish:
                    self._exec_finish_job(job_list)
                    break
                for job in job_list:
                    if self.job_iter_dict.get(job.get_job_id()) is not None \
                            and self.job_iter_dict.get(job.get_job_id()) >= job.get_iterations():
                        continue
                    aggregat_file, fed_step = self._find_latest_aggregate_model_pars(job.get_job_id())
                    if aggregat_file is not None and self.fed_step != fed_step:
                        if job.get_job_id() in runtime_config.EXEC_JOB_LIST:
                            runtime_config.EXEC_JOB_LIST.remove(job.get_job_id())
                        self.fed_step = fed_step

                    if job.get_job_id() not in runtime_config.EXEC_JOB_LIST:
                        job_model = self._load_job_model(job.get_job_id(), job.get_train_model_class_name())
                        if aggregat_file is not None:
                            print("load {} parameters".format(aggregat_file))
                            job_model.load_state_dict(torch.load(aggregat_file))
                        job_models_path = self._create_job_models_dir(self.client_id, job.get_job_id())
                        future = self.trainer_executor_pool.submit(self._train, self.data, job, job_model,
                                                                   job_models_path, self.fed_step)
                        runtime_config.EXEC_JOB_LIST.append(job.get_job_id())
                        print("job_{} is start training".format(job.get_job_id()))
                        future.result()
                        self.job_iter_dict[job.get_job_id()] += 1

                time.sleep(5)
        else:
            #TODO: MPC support
            pass


    def _train(self, data, job, train_model, job_models_path, fed_step):
        train_strategy = job.get_train_strategy()

        dataloader = torch.utils.data.DataLoader(data, batch_size=train_strategy.get_batch_size(), shuffle=True, num_workers=1,
                                           pin_memory=True)

        optimizer = self._parse_optimizer(train_strategy.get_optimizer(), train_model, train_strategy.get_learning_rate())
        for idx, (batch_data, batch_target) in enumerate(dataloader):
            batch_data, batch_target = batch_data.cuda(), batch_target.cuda()
            train_model = train_model.cuda()
            pred = train_model(batch_data)
            loss_function = self._parse_loss_function(train_strategy.get_loss_function(), pred, batch_target)
            optimizer.zero_grad()
            loss_function.backward()
            optimizer.step()
            if idx % 200 == 0:
                print("loss: ", loss_function.item())

        torch.save(train_model.state_dict(), job_models_path+"\\tmp_parameters_{}".format(fed_step))



    def _parse_optimizer(self, optimizer, model, lr):
        if optimizer == RunTimeStrategy.OPTIM_SGD:
            return torch.optim.SGD(model.parameters(), lr, momentum=True)


    def _parse_loss_function(self, loss_function, output, label):
        if loss_function == RunTimeStrategy.NLL_LOSS:
            loss = F.nll_loss(output, label)
        return loss



    def _list_all_jobs(self, job_path, job_iter_dict):
        job_list = []
        for file in os.listdir(job_path):
            # print("job file: ", job_path+"\\"+file)
            f = open(job_path+"\\"+file, "rb")
            job = pickle.load(f)
            job_list.append(job)
            if job_iter_dict.get(job.get_job_id()) is None:
                job_iter_dict[job.get_job_id()] = 0
        return job_list


    def _create_job_models_dir(self, client_id, job_id):
        # create local model dir
        local_model_dir = os.path.abspath(".") + "\\" + LOCAL_MODEL_BASE_PATH +"models_{}\\models_{}".format(job_id, client_id)
        if not os.path.exists(local_model_dir):
            os.makedirs(local_model_dir)
        return local_model_dir


    def _load_job_model(self, job_id, job_model_class_name):
        job_model_path = os.path.abspath(".") + "\\" + LOCAL_MODEL_BASE_PATH + "models_{}\\{}_init_model.py".format(job_id, job_id)
        module = importlib.import_module("res.models.models_{}.init_model_{}".format(job_id, job_id), "init_model_{}".format(job_id))
        model_class = getattr(module, job_model_class_name)
        return model_class()

    def _find_latest_aggregate_model_pars(self, job_id):
        job_model_path = LOCAL_MODEL_BASE_PATH + "models_{}\\{}".format(job_id, AGGREGATE_PATH)
        if not os.path.exists(job_model_path):
            os.makedirs(job_model_path)
        file_list = os.listdir(job_model_path)

        if len(file_list) != 0:
            return job_model_path + "\\" + file_list[-1], len(file_list)
        return None, 0


    def _prepare_jobs_model(self, job_list):
        for job in job_list:
            self._prepare_job_model(job)


    def _prepare_job_model(self, job):
        job_model_path = os.path.abspath(".") + "\\" + LOCAL_MODEL_BASE_PATH + "models_{}".format(job.get_job_id())
        job_init_model_path = job_model_path + "\\init_model_{}.py".format(job.get_job_id())
        model_f = open(job.get_train_model(), "r")
        if not os.path.exists(job_init_model_path):
            f = open(job_init_model_path, "w")
            for line in model_f.readlines():
                f.write(line)
            f.close()
        model_f.close()



    def _exec_finish_job(self, job_list):
        #TODO：acquire result and clean resources
        pass


