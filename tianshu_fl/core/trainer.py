import torch
import json
import torch.nn.functional as F
import time, os, pickle, requests, importlib, shutil, copy
from flask import url_for
from concurrent.futures import ThreadPoolExecutor
from tianshu_fl.core.strategy import WorkModeStrategy
from tianshu_fl.entity import runtime_config
from tianshu_fl.core.strategy import RunTimeStrategy
from tianshu_fl.core import communicate_client
from tianshu_fl.utils.utils import JobDecoder
from tianshu_fl.entity.job import Job


JOB_PATH = "res\\jobs"
LOCAL_MODEL_BASE_PATH = "res\\models\\"
AGGREGATE_PATH = "tmp_aggregate_pars"


class TrainStrategy(object):
    def __init__(self, client_id):
        self.client_id = client_id
        self.fed_step = {}
        self.job_iter_dict = {}
        self.job_path = os.path.abspath(".") + "\\" + JOB_PATH

    def _parse_optimizer(self, optimizer, model, lr):
        if optimizer == RunTimeStrategy.OPTIM_SGD:
            return torch.optim.SGD(model.parameters(), lr, momentum=0.5)

    def _compute_loss(self, loss_function, output, label):
        if loss_function == RunTimeStrategy.NLL_LOSS:
            loss = F.nll_loss(output, label)
        elif loss_function == RunTimeStrategy.KLDIV_LOSS:
            loss = F.kl_div(torch.log(output), label)
        return loss



    def _create_job_models_dir(self, client_id, job_id):
        # create local model dir
        local_model_dir = os.path.abspath(".") + "\\" + LOCAL_MODEL_BASE_PATH + "models_{}\\models_{}".format(job_id,
                                                                                                              client_id)
        if not os.path.exists(local_model_dir):
            os.makedirs(local_model_dir)
        return local_model_dir

    def _load_job_model(self, job_id, job_model_class_name):
        # job_model_path = os.path.abspath(".") + "\\" + LOCAL_MODEL_BASE_PATH + "models_{}\\{}_init_model.py".format(
        #     job_id, job_id)
        module = importlib.import_module("res.models.models_{}.init_model_{}".format(job_id, job_id),
                                         "init_model_{}".format(job_id))
        model_class = getattr(module, job_model_class_name)
        return model_class()



class TrainNormalStrategy(TrainStrategy):
    def __init__(self, job, data, fed_step, client_id):
        super(TrainNormalStrategy, self).__init__(client_id)
        self.job = job
        self.data = data
        self.job_model_path = os.path.abspath(".") + "\\models_{}".format(job.get_job_id())
        self.fed_step = fed_step


    def _train(self, train_model, job_models_path):
        train_strategy = self.job.get_train_strategy()

        dataloader = torch.utils.data.DataLoader(self.data, batch_size=train_strategy.get_batch_size(), shuffle=True,
                                                 num_workers=1,
                                                 pin_memory=True)

        optimizer = self._parse_optimizer(train_strategy.get_optimizer(), train_model,
                                          train_strategy.get_learning_rate())
        for idx, (batch_data, batch_target) in enumerate(dataloader):
            batch_data, batch_target = batch_data.cuda(), batch_target.cuda()
            train_model = train_model.cuda()
            pred = torch.log(train_model(batch_data))
            loss = self._compute_loss(train_strategy.get_loss_function(), pred, batch_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if idx % 200 == 0:
                print("loss: ", loss.item())

        torch.save(train_model.state_dict(), job_models_path + "\\tmp_parameters_{}".format(self.fed_step[self.job.get_job_id()]))

    def _exec_finish_job(self, job_list):
        pass



    def _find_latest_aggregate_model_pars(self, job_id):
        job_model_path = LOCAL_MODEL_BASE_PATH + "models_{}\\{}".format(job_id, AGGREGATE_PATH)
        if not os.path.exists(job_model_path):
            os.makedirs(job_model_path)
            init_model_pars_path = LOCAL_MODEL_BASE_PATH + "models_{}\\init_model_pars_{}".format(job_id, job_id)
            first_aggregate_path = LOCAL_MODEL_BASE_PATH + "models_{}\\tmp_aggregate_pars\\avg_pars_{}".format(job_id,
                                                                                                               0)
            if os.path.exists(init_model_pars_path):
                shutil.move(init_model_pars_path, first_aggregate_path)
        file_list = os.listdir(job_model_path)

        if len(file_list) != 0:
            return job_model_path + "\\" + file_list[-1], len(file_list)
        return None, 0

    def _prepare_jobs_model(self, job_list):
        for job in job_list:
            self._prepare_job_model(job)

    def _prepare_job_model(self, job):
        # prepare job model py file
        job_model_path = os.path.abspath(".") + "\\" + LOCAL_MODEL_BASE_PATH + "models_{}".format(job.get_job_id())
        job_init_model_path = job_model_path + "\\init_model_{}.py".format(job.get_job_id())
        with open(job.get_train_model(), "r") as model_f:
            if not os.path.exists(job_init_model_path):
                f = open(job_init_model_path, "w")
                for line in model_f.readlines():
                    f.write(line)
                f.close()





class TrainDistillationStrategy(TrainStrategy):
    def __init__(self, job, data, client_id):
        super(TrainDistillationStrategy, self).__init__(client_id)
        self.job = job
        self.data = data
        self.job_model_path = os.path.abspath(".")+"\\models_{}".format(job.get_job_id())


    def _load_other_models_pars(self, job_id, client_id):
        job_model_base_path = LOCAL_MODEL_BASE_PATH + "\\models_{}".format(job_id)
        other_models_pars = []
        for f in os.listdir(job_model_base_path):
            if f.find("models_") != -1 and int(f.split("_")[-1]) != client_id:
                files = os.listdir(f)
                if len(files) != 0:
                    other_models_pars.append(torch.load(files[-1]))
        return other_models_pars





class TrainStandloneNormalStrategy(TrainNormalStrategy):
    def __init__(self, job, data, fed_step, client_id):
        super(TrainStandloneNormalStrategy, self).__init__(job, data, fed_step, client_id)


    def train(self):
        # job_list = self._list_all_jobs(self.job_path, self.job_iter_dict)
        # self._prepare_jobs_model(job_list)
        # for job in job_list:
        #     if self.job_iter_dict.get(job.get_job_id()) is None or self.job_iter_dict.get(
        #             job.get_job_id()) != job.get_iterations():
        #         self.is_finish = False
        #         break
        # if self.is_finish:
        #     self._exec_finish_job(job_list)
        #     break
        # for job in job_list:
        self.job_iter_dict[self.job.get_job_id()] = 0 if self.job_iter_dict.get(self.job.get_job_id()) is None else self.job_iter_dict[self.job.get_job_id()]
        print("test_iter_num: ",self.job_iter_dict[self.job.get_job_id()])
        if self.job_iter_dict.get(self.job.get_job_id()) is not None \
                and self.job_iter_dict.get(self.job.get_job_id()) >= self.job.get_iterations():
            return
        aggregat_file, fed_step = self._find_latest_aggregate_model_pars(self.job.get_job_id())
        if aggregat_file is not None and self.fed_step.get(self.job.get_job_id()) != fed_step:
            if self.job.get_job_id() in runtime_config.EXEC_JOB_LIST:
                runtime_config.EXEC_JOB_LIST.remove(self.job.get_job_id())
            self.fed_step[self.job.get_job_id()] = fed_step
        if self.job.get_job_id() not in runtime_config.EXEC_JOB_LIST:
            job_model = self._load_job_model(self.job.get_job_id(), self.job.get_train_model_class_name())
            if aggregat_file is not None:
                print("load {} parameters".format(aggregat_file))
                job_model.load_state_dict(torch.load(aggregat_file))
            job_models_path = self._create_job_models_dir(self.client_id, self.job.get_job_id())
            print("job_{} is start training".format(self.job.get_job_id()))
            runtime_config.EXEC_JOB_LIST.append(self.job.get_job_id())
            self._train(job_model, job_models_path)
            self.job_iter_dict[self.job.get_job_id()] = fed_step





class TrainStandloneDistillationStrategy(TrainDistillationStrategy):
    def __init__(self, job, data, client_id):
        super(TrainStandloneDistillationStrategy, self).__init__(job, data, client_id)
        self.train_model = self._load_job_model(job.get_job_id(), job.get_train_model_class_name())
    def train(self):
        train_strategy = self.job.get_train_strategy()

        dataloader = torch.utils.data.DataLoader(self.data, batch_size=train_strategy.get_batch_size(), shuffle=True,
                                                 num_workers=1,
                                                 pin_memory=True)
        optimizer = self._parse_optimizer(train_strategy.get_optimizer(), self.train_model,
                                          train_strategy.get_learning_rate())

        other_models_pars = self._load_other_models_pars(self.job.get_job_id(), self.client_id)
        train_model, other_model = self.train_model.cuda(), copy.deepcopy(self.train_model)

        for idx, (batch_data, batch_target) in enumerate(dataloader):
            batch_data = batch_data.cuda()
            batch_target = batch_target.cuda()
            kl_pred = train_model(batch_data)
            pred = torch.log(kl_pred)

            loss_kl = self._compute_loss(RunTimeStrategy.KLDIV_LOSS, kl_pred, kl_pred)
            for other_model_pars in other_models_pars:
                other_model = other_model.load_state_dict(other_model_pars)
                other_model_kl_pred = other_model(batch_data).detach()
                loss_kl += self._compute_loss(RunTimeStrategy.KLDIV_LOSS, kl_pred, other_model_kl_pred)

            loss_s = self._compute_loss(train_strategy.get_loss_function(), pred, batch_target)
            loss = loss_s + self.job.get_distillation_alpha() * loss_kl
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if idx % 200 == 0:
                print("loss: ", loss.item())



class TrainMPCNormalStrategy(TrainNormalStrategy):
    def __init__(self, job, data, fed_step, client_ip, client_port, server_url,  client_id):
        super(TrainMPCNormalStrategy, self).__init__(job, data, fed_step, client_id)
        self.server_url = server_url
        self.client_ip = client_ip
        self.client_port = client_port

    def _prepare_job_init_model_pars(self, job, server_url):
        job_init_model_pars_dir = os.path.abspath(".") + "\\" + LOCAL_MODEL_BASE_PATH + \
                                  "models_{}\\tmp_aggregate_pars".format(job.get_job_id())
        if len(os.listdir(job_init_model_pars_dir)) == 0:
            # print("/".join([server_url, "modelpars", job.get_job_id()]))
            response = requests.get("/".join([server_url, "modelpars", job.get_job_id()]))
            with open(job_init_model_pars_dir + "\\avg_pars_0", "wb") as f:
                for chunck in response.iter_content(chunk_size=512):
                    if chunck:
                        f.write(chunck)

    def _prepare_upload_client_model_pars(self, job_id, client_id, fed_avg):
        job_init_model_pars_dir = os.path.abspath(".") + "\\" + LOCAL_MODEL_BASE_PATH + \
                                  "models_{}\\models_{}".format(job_id, client_id)
        tmp_parameter_path = "tmp_parameters_{}".format(fed_avg)

        files = {
            'tmp_parameter_file': (
                'tmp_parameter_file', open(job_init_model_pars_dir + "\\" + tmp_parameter_path, "rb"))
        }
        return files



    def train(self):
        while True:
            response = requests.get("/".join([self.server_url, "jobs"]))

            response_data = response.json()
            job_list_str = response_data['data']
            print(job_list_str)
            for job_str in job_list_str:
                job = json.loads(job_str, cls=JobDecoder)
                self._prepare_job_model(job)
                self._prepare_job_init_model_pars(job, self.server_url)
                aggregat_file, fed_step = self._find_latest_aggregate_model_pars(job.get_job_id())
                if aggregat_file is not None and self.fed_step.get(job.get_job_id()) != fed_step:
                    job_models_path = self._create_job_models_dir(self.client_id, job.get_job_id())
                    job_model = self._load_job_model(job.get_job_id(), job.get_train_model_class_name())
                    job_model.load_state_dict(torch.load(aggregat_file))
                    self.fed_step[job.get_job_id()] = fed_step
                    self._train(self.data, job_model)
                    files = self._prepare_upload_client_model_pars(job.get_job_id(), self.client_id,
                                                                   self.fed_step.get(job.get_job_id()))
                    response = requests.post("/".join(
                        [self.server_url, "modelpars", "%s" % self.client_id, "%s" % job.get_job_id(),
                         "%s" % self.fed_step[job.get_job_id()]]),
                                             data=None, files=files)
                    print(response)

            time.sleep(5)

class TrainMPCDistillationStrategy(TrainDistillationStrategy):
    def __init__(self, job, data, client_ip, client_port, server_url, client_id):
        super(TrainDistillationStrategy, self).__init__(job, data, client_id)
        self.client_ip = client_ip
        self.client_port = client_port
        self.server_url = server_url


    def train(self):
        pass