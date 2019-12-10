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




class TrainStrategy(object):
    def __init__(self, job, data, train_model, job_model_path):
        self.job = job
        self.data = data
        self.train_mdoel = train_model
        self.job_model_path = job_model_path


    def train(self):
        pass



class TrainStandloneNormalStrategy(TrainStrategy):
    def __init__(self, job, data, train_model, job_model_path, fed_step):
        super(TrainStandloneNormalStrategy, self).__init__(job, data, train_model, job_model_path)
        self.fed_step = fed_step


    def train(self):
        train_strategy = self.job.get_train_strategy()

        dataloader = torch.utils.data.DataLoader(self.data, batch_size=train_strategy.get_batch_size(), shuffle=True,
                                                 num_workers=1,
                                                 pin_memory=True)

        optimizer = self._parse_optimizer(train_strategy.get_optimizer(), self.train_model,
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

        torch.save(train_model.state_dict(), self.job_models_path + "\\tmp_parameters_{}".format(self.fed_step))

class TrainStandloneDistillationStrategy(TrainStrategy):
    def __init__(self, job, data, train_model, job_model_path, alpha):
        super(TrainStandloneDistillationStrategy, self).__init__(job, data, train_model, job_model_path)
        self.alpha = alpha

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
            loss = loss_s + self.alpha * loss_kl
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if idx % 200 == 0:
                print("loss: ", loss.item())



class TrainMPCNormalStrategy(TrainStrategy):
    def __init__(self):
        super(TrainMPCNormalStrategy, self).__init__()

    def train(self):
        pass

class TrainMPCDistillationStrategy(TrainStrategy):
    def __init__(self):
        super(TrainMPCDistillationStrategy, self).__init__()

    def train(self):
        pass