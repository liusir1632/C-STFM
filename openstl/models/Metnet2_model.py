import torch
import torch.nn as nn

from openstl.modules import ConvLSTMCell
from openstl.utils import (reduce_tensor, reshape_patch,
                           reserve_schedule_sampling_exp, schedule_sampling)
from openstl.modules.layers.DilatedCondConv import DilatedResidualConv
from openstl.modules.layers.ConditionWithTimeMetNet2 import ConditionWithTimeMetNet2

class metnet2_model(nn.Module):
    r"""ConvLSTM Model

    Implementation of `Convolutional LSTM Network: A Machine Learning Approach
    for Precipitation Nowcasting <https://arxiv.org/abs/1506.04214>`_.

    """

    def __init__(self, num_layers, num_hidden, configs, **kwargs):
        super(metnet2_model, self).__init__()
        T, C, H, W = configs.in_shape

        self.configs = configs
        self.frame_channel = configs.patch_size * configs.patch_size * C
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        cell_list = []

        height = H // configs.patch_size
        width = W // configs.patch_size
        self.MSE_criterion = nn.MSELoss()

        for i in range(num_layers):
            in_channel = self.frame_channel if i == 0 else num_hidden[i - 1]
            cell_list.append(
                ConvLSTMCell(in_channel, num_hidden[i], height, width, configs.filter_size,
                                       configs.stride, configs.layer_norm)
            )
        self.cell_list = nn.ModuleList(cell_list)
        self.conv_last = nn.Conv2d(num_hidden[num_layers - 1], self.frame_channel,
                                   kernel_size=1, stride=1, padding=0, bias=False)
        
        self.conv_next = nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0, bias=False)  
                # Shallow network of Conv Residual Block Dilation 1 with the lead time MLP embedding added
        
        encoder_dilations = (1, 2, 4, 8, 16, 32, 64, 128)
        num_context_blocks: int = 3
        lead_time_features: int = 2048
        
        
        self.context_block_one = nn.ModuleList()
        self.context_block_one.append(
            DilatedResidualConv(
                input_channels=32,
                output_channels=64,
                kernel_size=3,
                dilation=1,
            )
        )
        self.context_block_one.extend(
            [
                DilatedResidualConv(
                    input_channels=64,
                    output_channels=64,
                    kernel_size=3,
                    dilation=d,
                )
                for d in encoder_dilations[1:]
            ]
        )
        self.context_blocks = nn.ModuleList()
        for block in range(num_context_blocks - 1):
            self.context_blocks.extend(
                nn.ModuleList(
                    [
                        DilatedResidualConv(
                            input_channels=64,
                            output_channels=64,
                            kernel_size=3,
                            dilation=d,
                        )
                        for d in encoder_dilations
                    ]
                )
            )
        
        self.residual_block_three = nn.ModuleList(
            [
                DilatedResidualConv(
                    input_channels=1,
                    output_channels=1,
                    kernel_size=3,
                    dilation=1,
                )
                for _ in range(8)
            ]
        )

        self.time_conditioners = nn.ModuleList()
        # Go through each set of blocks and add conditioner
        # Context Stack
        for layer in self.context_block_one:
            self.time_conditioners.append(
                ConditionWithTimeMetNet2(
                    forecast_steps=23,
                    hidden_dim=lead_time_features,
                    num_feature_maps=layer.output_channels,
                )
            )
        for layer in self.context_blocks:
            self.time_conditioners.append(
                ConditionWithTimeMetNet2(
                    forecast_steps=23,
                    hidden_dim=lead_time_features,
                    num_feature_maps=layer.output_channels,
                )
            )

        for layer in self.residual_block_three:
            self.time_conditioners.append(
                ConditionWithTimeMetNet2(
                    forecast_steps=23,
                    hidden_dim=lead_time_features,
                    num_feature_maps=layer.output_channels,
                )
            )

    def reshape_patch_back(self, patch_tensor, patch_size):
        # [B, 8, 8 ,64]
        batch_size, patch_height, patch_width, channels = patch_tensor.shape
        img_channels = channels // (patch_size*patch_size)
        a = patch_tensor.reshape(batch_size,
                                    patch_height, patch_width,
                                    patch_size, patch_size,
                                    img_channels)
        b = a.transpose(2, 3)
        img_tensor = b.reshape(batch_size,
                                    patch_height * patch_size,
                                    patch_width * patch_size,
                                    img_channels)
        return img_tensor


    def next_deal(self, res, lead_time):
        block_num = 0
        # print("1-8", res.shape)         # torch.Size([B, 32, 16, 16])             [B, 32, 8, 8]

        # Context Stack
        for layer in self.context_block_one:
            scale, bias = self.time_conditioners[block_num](res, lead_time)
            res = layer(res, scale, bias)
            block_num += 1
        # print("1-9", res.shape)         # torch.Size([B, 64, 16, 16])             [B, 64, 8, 8]
        
        for layer in self.context_blocks:
            scale, bias = self.time_conditioners[block_num](res, lead_time)
            res = layer(res, scale, bias)
            block_num += 1
        # print("1-10", res.shape)        # torch.Size([B, 64, 16, 16])             [B, 64, 8, 8]
        
        res = res.permute(0, 2, 3, 1).contiguous()                                  # [B, 8, 8 ,64]
        res = self.reshape_patch_back(res, self.configs.patch_size)                 # 要修改 reshape_patch_back 函数，因为原本传入是5维 
                                                                                    # 应该变为 [B, 64, 64 ,1]
        
        res = res.permute(0, 3, 1, 2).contiguous()                                  # [B, 1, 64, 64]
        
        for layer in self.residual_block_three:
            scale, bias = self.time_conditioners[block_num](res, lead_time)
            res = layer(res, scale, bias)
            block_num += 1
        # print("1-13", res.shape)        # torch.Size([B, 128, 64, 64])            [B, 1, 64, 64]

        # Return 1x1 Conv
        # res = self.head(res)            # 这一步之后的输出维度 1 是 metnet2 代码也最上面的 output_channels: int = 12,
        # print("1-14", res.shape)       # torch.Size([B, 1, 64, 64])

        return res


    def forward(self, frames_tensor, mask_true, **kwargs):
        
        # 传进来的 frames_tensor 是分过patch的, 应该是[batch, 23, 8, 8, 64]
        
        # [batch, length, height, width, channel] -> [batch, length, channel, height, width]
        frames = frames_tensor.permute(0, 1, 4, 2, 3).contiguous()
        mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()

        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]

        next_frames = []
        h_t = []
        c_t = []

        for i in range(self.num_layers):
            zeros = torch.zeros([batch, self.num_hidden[i], height, width]).to(self.configs.device)
            h_t.append(zeros)
            c_t.append(zeros)
        
        for t in range(self.configs.pre_seq_length + self.configs.aft_seq_length - 1):
            # reverse schedule sampling
            if self.configs.reverse_scheduled_sampling == 1:
                if t == 0:
                    net = frames[:, t]
                else:
                    net = mask_true[:, t - 1] * frames[:, t] + (1 - mask_true[:, t - 1]) * x_gen
            else:
                if t < self.configs.pre_seq_length:
                    net = frames[:, t]
                else:
                    net = mask_true[:, t - self.configs.pre_seq_length] * frames[:, t] + \
                          (1 - mask_true[:, t - self.configs.pre_seq_length]) * x_gen

            h_t[0], c_t[0] = self.cell_list[0](net, h_t[0], c_t[0])

            for i in range(1, self.num_layers):
                h_t[i], c_t[i] = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i])

            x_gen = self.conv_last(h_t[self.num_layers - 1])
            next_frames.append(x_gen)                                   # 这里的 x_gen 的shape应该是 [B, 64, 8, 8]

        # [length, batch, channel, height, width] -> [batch, length, height, width, channel]
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 3, 4, 2).contiguous()
        # print('next_frames.shape', next_frames.shape)                   # 分了patch之后的预测的后23帧 [batch, length, height, width, channel]
                                                                        # 应该是 [B, 23, 8, 8, 64]
                                                                        
        # print('frames_tensor[:, 1:]', frames_tensor[:, 1:].shape)       # 分了patch之后的输入的后23帧 [batch, length, height, width, channel]
                                                                        # 应该是 [B, 23, 8, 8, 64]
                                                                        
        # [batch, length, height, width, channel] -> [batch, length, channel, height, width]
        next_frames = next_frames.permute(0, 1, 4, 2, 3).contiguous()   # 应该是 [B, 23, 64, 8, 8]
        
        # next_frames = torch.tensor(next_frames)

        # 按照时间维度进行解绑
        cur_layer_input = torch.unbind(next_frames, dim=1)
        seq_len = len(cur_layer_input)
        # print('seq_len', seq_len)                   # seq_len 23
        output_next = []
        for t in range(seq_len):
            # print(t)
            cur_input = cur_layer_input[t]
            # print('cur_input_1', cur_input.shape)     # torch.Size([B, 64, 8, 8])
            cur_input = self.conv_next(cur_input)
            # print('cur_input_2', cur_input.shape)     # torch.Size([B, 32, 8, 8])
            cur_input = self.next_deal(cur_input, t)
            output_next.append(cur_input)
        
        final_output = torch.stack(output_next, dim=1)
        # print('final_output.shape', final_output.shape)     # torch.Size([B, 23, 1, 64, 64])        
        
        # [batch, length, channel, height, width] -> [batch, length, height, width, channel]
        final_output = final_output.permute(0, 1, 3, 4, 2).contiguous()
        # 分patch的输入要求：[batch, length, height, width, channel]
        final_output = reshape_patch(final_output, self.configs.patch_size)     # torch.Size([B, 23, 8, 8, 64])
        
        # print('final_output.shape', final_output.shape)               # 应该是 [B, 23, 8, 8, 64]
        # print('frames_tensor[:, 1:]', frames_tensor[:, 1:].shape)       # 分了patch之后的输入的后23帧 [batch, length, height, width, channel]
                                                                        # 应该是 [B, 23, 8, 8, 64]
        if kwargs.get('return_loss', True):
            loss = self.MSE_criterion(final_output, frames_tensor[:, 1:])
        else:
            loss = None

        # final_output 的shape应该是 [B, 23, 8, 8, 64] 
        return final_output, loss
