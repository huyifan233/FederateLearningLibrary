
import time, os, pickle
import torch
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor
from tianshu_fl.core.strategy import WorkModeStrategy
from tianshu_fl.entity import runtime_config
from tianshu_fl.core.strategy import RunTimeStrategy

JOB_PATH = "res\\jobs"

class Trainer(object):
    def __init__(self, work_mode, data, concurrent_num=3):
        self.work_mode = work_mode
        self.data = data
        self.concurrent_num = concurrent_num
        self.trainer_executor_pool = ThreadPoolExecutor(self.concurrent_num)
        self.job_path = os.path.abspath(".")+"\\"+JOB_PATH

    def start(self):
        if self.work_mode == WorkModeStrategy.WORKMODE_STANDALONE:
            while True:
                job_file_list = Trainer.list_all_jobs(self.job_path)
                if job_file_list is not None:
                    print("len: {}".format(len(job_file_list)))
                    for job_file in job_file_list:
                        job = pickle.load(job_file)
                        self.trainer_executor_pool.submit(Trainer.train, self.data, job)
                    #TODO: need to send model to server and get terminate signal
                time.sleep(5)

    @staticmethod
    def train(self, data, job):
        train_strategy = job.get_train_strategy()
        dataloader = torch.utils.data.Dataloader(data, batch_size=train_strategy.get_batch_size(), shuffle=True, num_workers=1,
                                           pin_memory=True)
        train_model = job.get_train_model()
        optimizer = Trainer.parse_optimizer(train_strategy.get_optimizer(), train_model, train_strategy.get_learning_rate())
        for idx, (data, target) in enumerate(dataloader):
            data, target = data.cuda(), target.cuda()
            pred = train_model(data)
            loss_function = Trainer.parse_loss_function(train_strategy.get_loss_function(), pred, target)
            optimizer.zero_grad()
            loss_function.backward()
            optimizer.step()
            if idx % 100 == 0:
                print("loss: ", loss_function.item())



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
        file_list = []
        for file in os.listdir(job_path):
            f = open(job_path+"\\"+file, "rb")
            file_list.append(f)
        return file_list