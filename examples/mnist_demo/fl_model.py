import os
from torch import nn
import torch.nn.functional as F
import tianshu_fl.core.strategy as strategy
from tianshu_fl.core.job_manager import JobManager

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
    job_manager = JobManager(JOB_PATH)
    job = job_manager.generate_job(strategy.WorkModeStrategy.WORKMODE_STANDALONE, train_code_strategy, Net, 3)

    job_manager.submit_job(job, model, MODEL_PATH)
