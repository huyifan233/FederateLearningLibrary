
#federate strategies

import tianshu_fl.exceptions.fl_expection as exceptions


class WorkModeStrategy(object):
    WORKMODE_STANDALONE = "standalone"
    WORKMODE_CLUSTER = "cluster"



class FedrateStrategy(object):
    FED_AVG = "fed_avg"
    FED_DISTILLATION = "fed_distillation"



class RunTimeStrategy(object) :
    L1_LOSS = "L1loss"
    MSE_LOSS = "MSELoss"
    CROSSENTROPY_LOSS = "CrossEntropyLoss"
    NLL_LOSS = "NLLLoss"
    POISSIONNLL_LOSS = "PoissonNLLLoss"
    KLDIV_LOSS = "KLDivLoss"
    BCE_LOSS = "BCELoss"
    BCEWITHLOGITS_Loss = "BCEWithLogitsLoss"
    MARGINRANKING_Loss = "MarginRankingLoss"
    OPTIM_SGD = "SGD"
    OPTIM_ADAM = "Adam"


class StrategyFactory(object):
    def __init__(self):
        pass

class TrainStrategyFatorcy(StrategyFactory):

    def __init__(self):
        super(StrategyFactory, self).__init__()
        self.optimizer = None
        self.learning_rate = None
        self.loss_function = None
        self.batch_size = None
        self.epoch = None

    def get_loss_functions(self):
        loss_functions = [RunTimeStrategy.L1_LOSS, RunTimeStrategy.MSE_LOSS, RunTimeStrategy.CROSSENTROPY_LOSS, RunTimeStrategy.NLL_LOSS, RunTimeStrategy.POISSIONNLL_LOSS,
                          RunTimeStrategy.KLDIV_LOSS, RunTimeStrategy.BCE_LOSS, RunTimeStrategy.BCEWITHLOGITS_Loss, RunTimeStrategy.MARGINRANKING_Loss]
        return loss_functions


    def get_fed_strategies(self):
        fed_strategies = [FedrateStrategy.FED_AVG, FedrateStrategy.FED_DISTILLATION]
        return fed_strategies

    def get_optim_strategies(self):
        optim_strategies = [RunTimeStrategy.OPTIM_SGD, RunTimeStrategy.OPTIM_ADAM]
        return optim_strategies


    def set_optimizer(self, optimizer):
        optim_strategies = self.get_optim_strategies()
        if optimizer in optim_strategies:
            self.optimizer = optimizer
        else:
            raise exceptions.TianshuFLException("optimizer strategy not found")

    def get_optimizer(self):
        return self.optimizer

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate

    def get_learning_rate(self):
        return self.learning_rate

    def set_loss_function(self, loss_function):
        loss_functions = self.get_loss_functions()
        if loss_function in loss_functions:
            self.loss_function = loss_function
        else:
            raise exceptions.TianshuFLException("loss strategy not found")

    def get_loss_function(self):
        return self.loss_function

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def get_batch_size(self):
        return self.batch_size

    def set_epoch(self, epoch):
        self.epoch = epoch

    def get_epoch(self):
        return self.epoch


class TestStrategyFactory(StrategyFactory):

    def __init__(self):
        super(TestStrategyFactory, self).__init__()