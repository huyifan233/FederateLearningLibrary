import tianshu_fl.core.strategy as strategy
import tianshu_fl.exceptions.fl_expection as exceptions




class Code(object):
    def __init__(self):
        pass




class TrainCode(Code):

    def __init__(self):
        super(TrainCode, self).__init__()
        self.optimizer = None
        self.learning_rate = None
        self.loss_function = None
        self.batch_size = None
        self.epoch = None

    def set_optimizer(self, optimizer):
        optim_strategies = strategy.StrategyFatorcy.get_optim_strategies()
        if optimizer in optim_strategies:
            self.optimizer = optimizer
        else:
            raise exceptions.TianshuFLException("optimizer strategy not found")

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate

    def set_loss_function(self, loss_function):
        loss_functions = strategy.StrategyFatorcy.get_loss_functions()
        if loss_function in loss_functions:
            self.loss_function = loss_function
        else:
            raise exceptions.TianshuFLException("loss strategy not found")
    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def set_epoch(self, epoch):
        self.epoch = epoch


class TestCode(Code):
    def __init__(self):
        super(TestCode, self).__init__()