import torch.nn as nn

from openstl.models.Metnet2_model import metnet2_model
from .predrnn import PredRNN


class Metnet2(PredRNN):
    r"""ConvLSTM
    """

    def __init__(self, args, device, steps_per_epoch):
        PredRNN.__init__(self, args, device,  steps_per_epoch)
        self.model = self._build_model(self.args)
        self.model_optim, self.scheduler, self.by_epoch = self._init_optimizer(steps_per_epoch)
        self.criterion = nn.MSELoss()

    def _build_model(self, args):
        num_hidden = [int(x) for x in self.args.num_hidden.split(',')]
        num_layers = len(num_hidden)
        return metnet2_model(num_layers, num_hidden, args).to(self.device)


