import torch
import torch.nn as nn
import torch.nn.functional as F


class tf_Conv3d(nn.Module):

    def __init__(self, in_channels, out_channels, *vargs, **kwargs):
        super(tf_Conv3d, self).__init__()
        self.conv3d = nn.Conv3d(in_channels, out_channels, *vargs, **kwargs)

    def forward(self, input):
        return F.interpolate(self.conv3d(input), size=input.shape[-3:], mode="nearest")


class Eidetic3DLSTMCell(nn.Module):

    def __init__(self, in_channel, num_hidden, window_length,
                 height, width, filter_size, stride, layer_norm):
        super(Eidetic3DLSTMCell, self).__init__()

        self._norm_c_t = nn.LayerNorm([num_hidden, window_length, height, width])
        self.num_hidden = num_hidden
        self.padding = (0, filter_size[1] // 2, filter_size[2] // 2) 
        self._forget_bias = 1.0
        if layer_norm:
            self.conv_x = nn.Sequential(
                tf_Conv3d(in_channel, num_hidden * 7, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 7, window_length, height, width])
            )
            self.conv_h = nn.Sequential(
                tf_Conv3d(num_hidden, num_hidden * 4, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 4, window_length, height, width])
            )
            self.conv_gm = nn.Sequential(
                tf_Conv3d(num_hidden, num_hidden * 4, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 4, window_length, height, width])
            )
            self.conv_new_cell = nn.Sequential(
                tf_Conv3d(num_hidden, num_hidden, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden, window_length, height, width])
            )
            self.conv_new_gm = nn.Sequential(
                tf_Conv3d(num_hidden, num_hidden, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden, window_length, height, width])
            )
        else:
            self.conv_x = nn.Sequential(
                tf_Conv3d(in_channel, num_hidden * 7, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
            )
            self.conv_h = nn.Sequential(
                tf_Conv3d(num_hidden, num_hidden * 4, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
            )
            self.conv_gm = nn.Sequential(
                tf_Conv3d(num_hidden, num_hidden * 4, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
            )
            self.conv_new_cell = nn.Sequential(
                tf_Conv3d(num_hidden, num_hidden, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
            )
            self.conv_new_gm = nn.Sequential(
                tf_Conv3d(num_hidden, num_hidden, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
            )
        self.conv_last = tf_Conv3d(num_hidden * 2, num_hidden, kernel_size=1,
                                   stride=1, padding=0, bias=False)
    
    def _attn(self, in_query, in_keys, in_values):
        # 从in_query张量的形状中提取了五个值，分别是batch、num_channels（通道数）、_（暂时不使用，可能是通道数的维度）、width、height
        batch, num_channels, _, width, height = in_query.shape
        
        # 重新塑造（reshape）了in_query张量，使其具有形状 (batch, -1, num_channels)。
        # 这实际上是将原始张量转换成一个二维张量，其中第一个维度是批次大小，第二个维度是自动计算的以容纳剩余的数据，最后一个维度是通道数
        query = in_query.reshape(batch, -1, num_channels)
        
        # 重新塑造了in_keys张量，使其具有形状 (batch, -1, num_channels)。
        # 这是为了将in_keys的数据整理成与in_query相同的形状，以便进行注意力计算
        keys = in_keys.reshape(batch, -1, num_channels)
        
        # 同上
        values = in_values.reshape(batch, -1, num_channels)
        
        # 使用torch.einsum函数执行了一个张量乘法操作。具体来说，它执行了一个批次内的点积操作
        # 将query和keys的维度'bxc'和'byc'合并成'bxy'，得到一个形状为(batch, num_channels, width, height)的attn张量。这个张量包含了注意力分数
        attn = torch.einsum('bxc,byc->bxy', query, keys)
        
        # 使用torch.softmax函数在第2个维度（'dim=2'，也就是宽度和高度维度）上计算了attn张量的softmax，将其规范化为概率分布。
        # 这使得attn张量的值在每个位置上都变成了0到1之间的值，表示了每个位置的重要性
        attn = torch.softmax(attn, dim=2)
        
        # 再次使用torch.einsum函数，执行了一个张量乘法操作。它将attn与values的维度'bxy'和'byc'合并成'bxc'
        # 得到一个新的张量attn，其中存储了根据注意力分布加权的值
        attn = torch.einsum("bxy,byc->bxc", attn, values)
        
        # 将attn张量重新塑造成原始形状，其中第一个维度是批次大小，第二个维度是通道数，然后剩余的维度是重新排列的原始宽度和高度。
        # 返回这个最终的attn张量，它包含了经过注意力机制计算后的值
        return attn.reshape(batch, num_channels, -1, width, height)

    def forward(self, x_t, h_t, c_t, global_memory, eidetic_cell):
        h_concat = self.conv_h(h_t)
        i_h, g_h, r_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)

        x_concat = self.conv_x(x_t)
        i_x, g_x, r_x, o_x, temp_i_x, temp_g_x, temp_f_x = \
            torch.split(x_concat, self.num_hidden, dim=1)

        i_t = torch.sigmoid(i_x + i_h)
        r_t = torch.sigmoid(r_x + r_h)
        g_t = torch.tanh(g_x + g_h)

        new_cell = c_t + self._attn(r_t, eidetic_cell, eidetic_cell)
        new_cell = self._norm_c_t(new_cell) + i_t * g_t

        new_global_memory = self.conv_gm(global_memory)
        i_m, f_m, g_m, m_m = torch.split(new_global_memory, self.num_hidden, dim=1)

        temp_i_t = torch.sigmoid(temp_i_x + i_m)
        temp_f_t = torch.sigmoid(temp_f_x + f_m + self._forget_bias)
        temp_g_t = torch.tanh(temp_g_x + g_m)
        new_global_memory = temp_f_t * torch.tanh(m_m) + temp_i_t * temp_g_t
        
        o_c = self.conv_new_cell(new_cell)
        o_m = self.conv_new_gm(new_global_memory)

        output_gate = torch.tanh(o_x + o_h + o_c + o_m)

        memory = torch.cat((new_cell, new_global_memory), 1)
        memory = self.conv_last(memory)

        output = torch.tanh(memory) * torch.sigmoid(output_gate)

        return output, new_cell, global_memory
