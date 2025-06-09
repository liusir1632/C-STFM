import torch
import torch.nn as nn


class ConvLSTM8_Cell(nn.Module):

    def __init__(self, in_channel, num_hidden, height, width, filter_size, stride, layer_norm):
        super(ConvLSTM8_Cell, self).__init__()

        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0
        if layer_norm:
            self.conv_x = nn.Sequential(
                nn.Conv2d(in_channel, num_hidden * 4, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 4, height, width])
            )
            self.conv_h = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 4, height, width])
            )
            self.conv_o = nn.Sequential(
                nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden, height, width])
            )
        else:
            self.conv_x = nn.Sequential(
                nn.Conv2d(in_channel, num_hidden * 4, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
            )
            self.conv_h = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
            )
            self.conv_o = nn.Sequential(
                nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
            )
        self.conv_last = nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=1,
                                   stride=1, padding=0, bias=False)
    
        self._norm_c_t = nn.LayerNorm([num_hidden, height, width])
    
    
    def _attn(self, in_query, in_keys, in_values):
        batch, num_channels, height, width = in_query.shape  # 去掉了不需要的维度
        query = in_query.reshape(batch, num_channels, -1)  # 重新塑造 query 张量
        keys = in_keys.reshape(batch, num_channels, -1)     # 重新塑造 keys 张量
        values = in_values.reshape(batch, num_channels, -1) # 重新塑造 values 张量
        attn = torch.einsum('bxc,byc->bx', query, keys)
        attn = torch.softmax(attn, dim=1)
        attn = torch.einsum("bx,byc->bxc", attn, values)
        return attn.reshape(batch, num_channels, height, width)  # 返回时重新塑造成原始形状

    
    def forward(self, x_t, h_t, c_t, eidetic_cell):
        x_concat = self.conv_x(x_t)
        # x_concat = self.conv_x(x_t) + self._attn(x_t, c_t, c_t)
        h_concat = self.conv_h(h_t)
        i_x, f_x, g_x, o_x = torch.split(x_concat, self.num_hidden, dim=1)
        i_h, f_h, g_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)

        i_t = torch.sigmoid(i_x + i_h)
        f_t = torch.sigmoid(f_x + f_h)
        g_t = torch.tanh(g_x + g_h)

        c_new = c_t + self._attn(f_t, eidetic_cell, eidetic_cell)
        c_new = self._norm_c_t(c_new) + i_t * g_t        
        
        # c_new = f_t * c_t + i_t * g_t

        o_t = torch.sigmoid(o_x + o_h)
        h_new = o_t * g_t
        return h_new, c_new
