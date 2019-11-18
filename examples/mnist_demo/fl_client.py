from tianshu_fl.core.strategy import WorkModeStrategy
from tianshu_fl.entity.runtime_config import RuntimeConfig
from tianshu_fl.core.trainer import Trainer


MODEL_PATH = "../mnist_res"

def start_trainer(work_mode):
    # if work_mode == WorkModeStrategy.WORKMODE_STANDALONE:
    #     pending_job_list = RuntimeConfig.PENDING_JOB_LIST
    #     trainer = Trainer()
    #     for pending_job in pending_job_list:
    #         trainer.train(pending_job)
    #
    # else:
    #     pass
    Trainer(work_mode, 3).start()


if __name__ == "__main__":

    start_trainer(WorkModeStrategy.WORKMODE_STANDALONE)










