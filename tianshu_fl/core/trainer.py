
import threading
import torch
from concurrent.futures import ThreadPoolExecutor
from tianshu_fl.core.strategy import WorkModeStrategy
from tianshu_fl.entity.runtime_config import RuntimeConfig
from tianshu_fl.core.strategy import RunTimeStrategy


class Trainer(object):
    def __init__(self, work_mode, concurrent_num=3):
        self.work_mode = work_mode
        self.concurrent_num = concurrent_num
        self.trainer_executor_pool = ThreadPoolExecutor(self.concurrent_num)


    def start(self):
        if self.work_mode == WorkModeStrategy.WORKMODE_STANDALONE:
            pending_job_list = RuntimeConfig.PENDING_JOB_LIST
            while True:
                for job in pending_job_list:
                    self.trainer_executor_pool.submit(Trainer.train, job)
                #TODO: need to send model to server and get terminate signal

    @staticmethod
    def train(self, job):
        optimizer =

    @staticmethod
    def parse_optimizer(self, optimizer, model):
        if optimizer == RunTimeStrategy.OPTIM_SGD:
            return torch.optim.SGD()


    def parse_loss_function(self):
        pass


