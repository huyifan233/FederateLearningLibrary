
import time, os, pickle
import torch
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor
from tianshu_fl.core.strategy import WorkModeStrategy
from tianshu_fl.entity import runtime_config
from tianshu_fl.core.strategy import RunTimeStrategy



JOB_PATH = "res\\jobs"
LOCAL_MODEL_BASE_PATH = "res\\models\\"
AGGREGATE_PATH = "tmp_pars"


class Trainer(object):
    def __init__(self, work_mode, data, client_id, concurrent_num=3):
        self.work_mode = work_mode
        self.data = data
        self.client_id = client_id
        self.concurrent_num = concurrent_num
        self.trainer_executor_pool = ThreadPoolExecutor(self.concurrent_num)
        self.job_path = os.path.abspath(".")+"\\"+JOB_PATH
        self.fed_step = 0

    def start(self):
        if self.work_mode == WorkModeStrategy.WORKMODE_STANDALONE:

            # start train
            while True:
                job_list = Trainer.list_all_jobs(self.job_path)
                for job in job_list:
                    aggregat_file, fed_step = Trainer.find_latest_aggregate_model_pars(job.get_job_id())
                    if aggregat_file is not None and self.fed_step != fed_step:
                        if job.get_job_id() in runtime_config.EXEC_JOB_LIST:
                            runtime_config.EXEC_JOB_LIST.remove(job.get_job_id())
                        self.fed_step = fed_step

                    if job.get_job_id() not in runtime_config.EXEC_JOB_LIST:
                        job_model = Trainer.load_job_model(job.get_job_id())
                        if aggregat_file is not None:
                            job_model.load_state_dict(torch.load(aggregat_file))
                        job_models_path = Trainer.create_job_models_dir(self.client_id, job.get_job_id())
                        future = self.trainer_executor_pool.submit(Trainer.train, self.data, job, job_model,
                                                                   job_models_path, self.fed_step)
                        runtime_config.EXEC_JOB_LIST.append(job.get_job_id())
                        print("job_{} is start training".format(job.get_job_id()))
                        future.result()

                time.sleep(5)
        else:
            #TODO: MPC support
            pass

    @staticmethod
    def train(data, job, train_model, job_models_path, fed_step):
        train_strategy = job.get_train_strategy()

        dataloader = torch.utils.data.DataLoader(data, batch_size=train_strategy.get_batch_size(), shuffle=True, num_workers=1,
                                           pin_memory=True)

        optimizer = Trainer.parse_optimizer(train_strategy.get_optimizer(), train_model, train_strategy.get_learning_rate())
        for idx, (batch_data, batch_target) in enumerate(dataloader):
            batch_data, batch_target = batch_data.cuda(), batch_target.cuda()
            train_model = train_model.cuda()
            pred = train_model(batch_data)
            loss_function = Trainer.parse_loss_function(train_strategy.get_loss_function(), pred, batch_target)
            optimizer.zero_grad()
            loss_function.backward()
            optimizer.step()
            if idx % 200 == 0:
                print("loss: ", loss_function.item())

        torch.save(train_model.state_dict(), job_models_path+"\\tmp_parameters_{}".format(fed_step))


    @staticmethod
    def parse_optimizer(optimizer, model, lr):
        if optimizer == RunTimeStrategy.OPTIM_SGD:
            return torch.optim.SGD(model.parameters(), lr, momentum=True)

    @staticmethod
    def parse_loss_function(loss_function, output, label):
        if loss_function == RunTimeStrategy.NLL_LOSS:
            loss = F.nll_loss(output, label)
        return loss


    @staticmethod
    def list_all_jobs(job_path):
        job_list = []
        for file in os.listdir(job_path):
            # print("job file: ", job_path+"\\"+file)
            f = open(job_path+"\\"+file, "rb")
            job = pickle.load(f)
            job_list.append(job)
        return job_list

    @staticmethod
    def create_job_models_dir(client_id, job_id):
        # create local model dir
        local_model_dir = os.path.abspath(".") + "\\" + LOCAL_MODEL_BASE_PATH +"models_{}\\models_{}".format(job_id, client_id)
        if not os.path.exists(local_model_dir):
            os.makedirs(local_model_dir)
        return local_model_dir

    @staticmethod
    def load_job_model(job_id):
        job_model_path = os.path.abspath(".") + "\\" + LOCAL_MODEL_BASE_PATH + "models_{}\\{}.pt".format(job_id, job_id)
        return torch.load(job_model_path)

    @staticmethod
    def find_latest_aggregate_model_pars(job_id):
        job_model_path = LOCAL_MODEL_BASE_PATH + "models_{}\\{}".format(job_id, AGGREGATE_PATH)
        file_list = os.listdir(job_model_path)

        if len(file_list) != 0:
            return job_model_path + "\\" + file_list[-1], len(file_list)
        return None, 0