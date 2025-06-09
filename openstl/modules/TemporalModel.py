import torch
from torch import Tensor
from torch import nn


class MulTemporalSelfAttention(nn.Module):

    def __init__(self, bias=False):
        super(MulTemporalSelfAttention, self).__init__()
        self.weight = nn.Linear(in_features=2, out_features=2, bias=bias)

    def forward(self, x_forward, x_backward) -> Tensor:
        x = torch.stack(tensors=[x_forward, x_backward], dim=2)
        x_T = x.permute(dims=[0, 1, 3, 2])
        weight_scores = self.weight(x_T)
        attention_scores = torch.softmax(weight_scores, dim=3)
        out = torch.mul(attention_scores, x_T)
        out = torch.sum(out, dim=3)
        return out


class MulTemporalSelfAttentionBasedLinearNet(nn.Module):
    def __init__(self, input_size: int = 1, hidden_size: int = 128, bias=False, activation_function=nn.SiLU(inplace=False),
                 isOut=False):
        super(MulTemporalSelfAttentionBasedLinearNet, self).__init__()
        self.isOut = isOut
        self.hidden_size = hidden_size
        self.encoder = nn.LSTM(input_size=input_size, hidden_size=hidden_size, bidirectional=True)
        self.temporal_attention = MulTemporalSelfAttention(bias=bias)
        self.linear = nn.Linear(in_features=hidden_size, out_features=input_size)
        self.activation_function = activation_function

    def forward(self, x, isOut):
        encoder_out, _ = self.encoder(x)
        x_forward = encoder_out[:, :, 0: self.hidden_size]
        x_backward = encoder_out[:, :, self.hidden_size:]
        if isOut:
            out = self.activation_function(
                self.linear(self.temporal_attention(x_forward, x_backward))
            )
        else:
            out = self.activation_function(
                self.temporal_attention(x_forward, x_backward)
            )
        return out


class TemporaNet(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, activation_fun=nn.SiLU(inplace=False),isOut=False,
                 bias=False):
        super(TemporaNet, self).__init__()
        self.isOut = isOut
        self.net = MulTemporalSelfAttentionBasedLinearNet(
            input_size=input_size, hidden_size=hidden_size, activation_function=activation_fun, bias=bias
        )

    def forward(self, x: Tensor):
        # x:[t, bhw, 1] -> out:[t, bhw, hidden_size]
        out = self.net(x, self.isOut)
        return out

