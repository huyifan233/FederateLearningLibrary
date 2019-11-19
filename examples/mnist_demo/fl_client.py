import torch
import os
from torchvision import datasets, transforms
from tianshu_fl.core.strategy import WorkModeStrategy
from tianshu_fl.core.trainer import Trainer
from tianshu_fl.core.job_detector import JobDetector



def start_trainer(work_mode, data):

    Trainer(work_mode, data, 3).start()
    #print(os.path.abspath("."))

if __name__ == "__main__":
    mnist_data = datasets.MNIST("./mnist_data", download=True, train=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.13066062,), (0.30810776,))
    ]))

    start_trainer(WorkModeStrategy.WORKMODE_STANDALONE, mnist_data)










