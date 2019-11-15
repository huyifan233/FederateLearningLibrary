
#federate strategies
FED_AVG = "fed_avg"
FED_DISTILLATION = "fed_distillation"
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
        loss_functions = [L1_LOSS, MSE_LOSS, CROSSENTROPY_LOSS, NLL_LOSS, POISSIONNLL_LOSS,
                          KLDIV_LOSS, BCE_LOSS, BCEWITHLOGITS_Loss, MARGINRANKING_Loss]
        return loss_functions

    @staticmethod
    def get_fed_strategies(self):
        fed_strategies = [FED_AVG, FED_DISTILLATION]
        return fed_strategies

    def get_optim_strategies(self):
        optim_strategies = [OPTIM_SGD, OPTIM_ADAM]
        return optim_strategies