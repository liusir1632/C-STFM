"""Originally adapted from https://github.com/aserdega/convlstmgru, MIT License Andriy Serdega"""
from typing import Any, List, Optional

import torch
import torch.nn as nn
from torch import Tensor


class ConvLSTMCell(nn.Module):
    """ConvLSTM Cell"""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        kernel_size: int,
        bias=True,
        activation=torch.tanh,
        batchnorm=False,
    ):
        """
        ConLSTM Cell

        Args:
            input_dim: Number of input channels
            hidden_dim: Number of hidden channels
            kernel_size: Kernel size
            bias: Whether to add bias
            activation: Activation to use
            batchnorm: Whether to use batch norm
        """
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size // 2, kernel_size // 2
        self.bias = bias
        self.activation = activation
        self.batchnorm = batchnorm

        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias,
        )

        self.reset_parameters()

    def forward(self, x: torch.Tensor, prev_state: list):
        """
        Compute forward pass

        Args:
            x: Input tensor of [Batch, Channel, Height, Width]
            prev_state: Previous hidden state

        Returns:
            The new hidden state and output
        """

        # 接受三个参数：self（表示类的实例对象，通常在类的方法中使用），x（输入数据的张量），和prev_state（先前的隐藏状态，以列表形式传入）

        # 将prev_state列表中的两个元素分别赋值给h_prev和c_prev，这些元素通常代表LSTM的先前隐藏状态和细胞状态
        h_prev, c_prev = prev_state

        # 将输入张量x和先前的隐藏状态h_prev沿着通道维度（dim=1）拼接在一起，创建一个新的张量combined
        # 这个步骤是为了将当前时间步的输入和先前时间步的隐藏状态合并起来，以便模型可以同时考虑它们
        combined = torch.cat((x, h_prev), dim=1)  # concatenate along channel axis
        combined_conv = self.conv(combined)

        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)

        g = self.activation(cc_g)
        c_cur = f * c_prev + i * g

        o = torch.sigmoid(cc_o)

        h_cur = o * self.activation(c_cur)

        return h_cur, c_cur

    def init_hidden(self, x: torch.Tensor):
        """
        Initializes the hidden state
        Args:
            x: Input tensor to initialize for

        Returns:
            Tuple containing the hidden states
        """
        state = (
            torch.zeros(x.size()[0], self.hidden_dim, x.size()[3], x.size()[4]),
            torch.zeros(x.size()[0], self.hidden_dim, x.size()[3], x.size()[4]),
        )
        state = (state[0].type_as(x), state[1].type_as(x))
        return state

    def reset_parameters(self):
        """Resets parameters"""
        nn.init.xavier_uniform_(self.conv.weight, gain=nn.init.calculate_gain("tanh"))
        self.conv.bias.data.zero_()

        if self.batchnorm:
            self.bn1.reset_parameters()
            self.bn2.reset_parameters()


class ConvLSTM(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        kernel_size: int,
        num_layers: int,
        bias=True,
        activation=torch.tanh,
        batchnorm=False,
    ):
        """
        ConvLSTM module

        Args:
            input_dim: Input dimension size
            hidden_dim: Hidden dimension size
            kernel_size: Kernel size
            num_layers: Number of layers
            bias: Whether to add bias
            activation: Activation function
            batchnorm: Whether to use batch norm
        """
        super(ConvLSTM, self).__init__()
        self.output_channels = hidden_dim
        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        activation = self._extend_for_multilayer(activation, num_layers)

        if not len(kernel_size) == len(hidden_dim) == len(activation) == num_layers:
            raise ValueError("Inconsistent list length.")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = True
        self.bias = bias

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(
                ConvLSTMCell(
                    input_dim=cur_input_dim,
                    hidden_dim=self.hidden_dim[i],
                    kernel_size=self.kernel_size[i],
                    bias=self.bias,
                    activation=activation[i],
                    batchnorm=batchnorm,
                )
            )

        self.cell_list = nn.ModuleList(cell_list)

        self.reset_parameters()

    def forward(
        self, x: torch.Tensor, hidden_state: Optional[list] = None
    ):
        """
        Computes the output of the ConvLSTM

        Args:
            x: Input Tensor of shape [Batch, Time, Channel, Width, Height]
            hidden_state: List of hidden states to use, if none passed, it will be generated

        Returns:
            The layer output and list of last states
        """
        # x表示输入数据，类型为torch.Tensor，hidden_state表示隐藏状态列表，类型为可选参数list。
        # 函数返回一个元组，包含两个元素，第一个元素是输出的Tensor，第二个元素是最后一层的隐藏状态列表。

        # 将输入数据x在指定维度（由self.batch_first决定）上解绑，得到一个包含时间步的列表cur_layer_input。
        # self.batch_first是一个控制是否批处理维度在第一维的布尔值。默认为True，则dim = 1
        cur_layer_input = torch.unbind(x, dim=int(self.batch_first))

        # 检查是否提供了隐藏状态。如果hidden_state为None，则表示没有提供隐藏状态，需要在下面的代码中生成初始状态。
        if not hidden_state:
            hidden_state = self.get_init_states(x)


        # 计算输入序列的长度，也就是时间步的数量
        seq_len = len(cur_layer_input)

        # 创建一个空列表last_state_list，用于存储每一层的最后一个时间步的隐藏状态
        last_state_list = []

        # 这是一个外循环，迭代每一层的ConvLSTM
        for layer_idx in range(self.num_layers):
            # 获取当前层的初始隐藏状态h和细胞状态c
            h, c = hidden_state[layer_idx]

            # 创建一个空列表output_inner，用于存储当前层每个时间步的输出
            output_inner = []

            # 这是一个内循环，迭代每个时间步
            for t in range(seq_len):
                # 在每个时间步，调用ConvLSTM的self.cell_list[layer_idx]层来计算新的隐藏状态h和细胞状态c，
                # 并将当前时间步的输入数据cur_layer_input[t]和前一时间步的状态[h, c]传递给该层。
                h, c = self.cell_list[layer_idx](x=cur_layer_input[t], prev_state=[h, c])
                # 将当前时间步的输出h添加到output_inner列表中
                output_inner.append(h)

            # 更新当前层的输入为output_inner，以便在下一个时间步中使用
            cur_layer_input = output_inner

            # 将当前层的最后一个时间步的隐藏状态(h, c)添加到last_state_list中
            last_state_list.append((h, c))

        # 在当前层的循环结束后，将output_inner列表中的输出堆叠起来，形成当前层的输出layer_output。堆叠的维度由self.batch_first决定。
        layer_output = torch.stack(output_inner, dim=int(self.batch_first))

        # 返回当前层的输出layer_output和所有层的最后隐藏状态的列表last_state_list作为函数的输出
        return layer_output, last_state_list

    def reset_parameters(self) -> None:
        """
        Reset parameters
        """
        for c in self.cell_list:
            c.reset_parameters()

    def get_init_states(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Constructs the initial hidden states

        Args:
            x: Tensor to use for constructing state

        Returns:
            The initial hidden states for all the layers in the network
        """
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(x))
        return init_states

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        """
        Extends a parameter for multiple layers

        Args:
            param: Parameter to copy
            num_layers: Number of layers

        Returns:
            The extended parameter
        """
        if not isinstance(param, list):
            param = [param] * num_layers
        return param
