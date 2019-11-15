import flask
import torch
from torch import nn
import torch.nn.functional as F
import tianshu_fl.generator.job_generator
import tianshu_fl.core.strategy as strategy
import tianshu_fl.generator.job_generator as job_generator
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

    def p_for_KL(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.softmax(x)
        return x

def generator_job(work_mode):

    job = job_generator.Job()
    job_id = job.generate_job_id()
    job.set_job_id(job_id)
    if work_mode == strategy.WorkModeStrategy.WORKMODE_STANDALONE:
        job.set_server_host("localhost:8080")
    else:
        job.set_server_host("");


if __name__ == "__main__":
    #startup(strategy.WorkModeStrategy.WORKMODE_STANDALONE)
    pass


