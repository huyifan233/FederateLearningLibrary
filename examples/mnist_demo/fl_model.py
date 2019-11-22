import flask
import torch
import pickle, os, inspect
from torch import nn
import torch.nn.functional as F
from tianshu_fl.generator.job_generator import Job
import tianshu_fl.core.strategy as strategy
from tianshu_fl.core.job_manager import JobManager

from tianshu_fl.generator.utils import JobUtils

JOB_PATH = os.path.abspath(".")+"\\res\\jobs"
MODEL_PATH = os.path.abspath(".")+"\\res\\models"

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    # def p_for_KL(self, x):
    #     x = F.relu(self.conv1(x))
    #     x = F.max_pool2d(x, 2, 2)
    #     x = F.relu(self.conv2(x))
    #     x = F.max_pool2d(x, 2, 2)
    #     x = x.view(-1, 4*4*50)
    #     x = F.relu(self.fc1(x))
    #     x = self.fc2(x)
    #     x = self.softmax(x)
    #     return x

def generator_job(work_mode, train_code_strategy, model, iter):

    job = Job()
    job.set_job_id(JobUtils.generate_job_id())
    if work_mode == strategy.WorkModeStrategy.WORKMODE_STANDALONE:
        job.set_server_host("localhost:8080")
    else:
        job.set_server_host("")
    job.set_train_strategy(train_code_strategy)
    job.set_train_model(inspect.getsource(model))
    job.set_iterations(iter)
    return job


def generate_train_strategy(optimizer, loss_function, lr=0.01, epoch=100, batch_size=32):

    train_code_strategy = strategy.TrainStrategyFatorcy()
    train_code_strategy.set_optimizer(optimizer)
    train_code_strategy.set_loss_function(loss_function)
    train_code_strategy.set_learning_rate(lr)
    train_code_strategy.set_epoch(epoch)
    train_code_strategy.set_batch_size(batch_size)
    return train_code_strategy

if __name__ == "__main__":
    #startup(strategy.WorkModeStrategy.WORKMODE_STANDALONE)
    train_code_strategy = generate_train_strategy(strategy.RunTimeStrategy.OPTIM_SGD, strategy.RunTimeStrategy.NLL_LOSS,
                                                  lr=0.01, epoch=100, batch_size=32)

    model = Net()
    job = generator_job(strategy.WorkModeStrategy.WORKMODE_STANDALONE, train_code_strategy, Net, 3)


    JobManager(JOB_PATH).submit_job(job, model, MODEL_PATH)
