import torch
import os
from torchvision import datasets, transforms
from tianshu_fl.core.strategy import WorkModeStrategy
from tianshu_fl.core.trainer_controller import Trainer
from tianshu_fl.core.job_detector import JobDetector
from torch import nn
import torch.nn.functional as F


SERVER_URL = "http://127.0.0.1:9763"
CLIENT_IP = "127.0.0.1"
CLIENT_PORT = 8081
CLIENT_ID = 0

def start_trainer(work_mode, client_id, data):
    if work_mode == WorkModeStrategy.WORKMODE_STANDALONE:
        Trainer(work_mode, data, client_id, None, None, 3).start()
    else:
        Trainer(work_mode, data, client_id, CLIENT_IP, CLIENT_PORT, SERVER_URL, 5).start()
    #print(os.path.abspath("."))

if __name__ == "__main__":
    mnist_data = datasets.MNIST("./mnist_data", download=True, train=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.13066062,), (0.30810776,))
    ]))

    #start_trainer(WorkModeStrategy.WORKMODE_STANDALONE, CLIENT_ID, mnist_data)

    start_trainer(WorkModeStrategy.WORKMODE_CLUSTER, CLIENT_ID, mnist_data)




