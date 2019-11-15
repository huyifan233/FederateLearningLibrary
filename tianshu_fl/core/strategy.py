
#federate strategies




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


class StrategyFatorcy(object):
    def __init__(self):
        pass
    @staticmethod
    def get_loss_functions(self):
        loss_functions = [RunTimeStrategy.L1_LOSS, RunTimeStrategy.MSE_LOSS, RunTimeStrategy.CROSSENTROPY_LOSS, RunTimeStrategy.NLL_LOSS, RunTimeStrategy.POISSIONNLL_LOSS,
                          RunTimeStrategy.KLDIV_LOSS, RunTimeStrategy.BCE_LOSS, RunTimeStrategy.BCEWITHLOGITS_Loss, RunTimeStrategy.MARGINRANKING_Loss]
        return loss_functions

    @staticmethod
    def get_fed_strategies(self):
        fed_strategies = [FedrateStrategy.FED_AVG, FedrateStrategy.FED_DISTILLATION]
        return fed_strategies

    def get_optim_strategies(self):
        optim_strategies = [RunTimeStrategy.OPTIM_SGD, RunTimeStrategy.OPTIM_ADAM]
        return optim_strategies