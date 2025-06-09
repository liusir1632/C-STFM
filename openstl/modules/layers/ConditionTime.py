import torch
from torch import nn as nn

# 该函数的作用是生成一个时间条件的热图（one-hot encoded time image），将时间信息添加到输入图像中
def condition_time(x, i=0, size=(12, 16), seq_len=15):
    """
    x: 输入图像，通常是一个张量。
    i: 时间步(默认为0), 表示要生成哪个时间步的条件热图。
    size: 生成的热图的大小，默认为(12, 16)。
    seq_len: 序列长度, 表示时间步的总数, 默认为15
    """
    # 断言，确保i小于seq_len，即所选择的时间步在合理范围内
    assert i < seq_len
    
    # 创建一个独热编码的时间热图。
    times = (torch.eye(seq_len, dtype=torch.long, device=x.device)[i]).unsqueeze(-1).unsqueeze(-1)
    
    # 创建一个与输入图像大小相同的全1张量，用于表示所有像素都具有时间条件信息。
    ones = torch.ones(1, *size, dtype=x.dtype, device=x.device)
    
    # 将时间热图与全1张量相乘，以生成最终的时间条件热图，并返回
    return times * ones


# 该函数的作用是在输入图像上添加时间条件信息, 结果是在输入数据的通道维度增加 时间步数 数量的通道
class ConditionTime(nn.Module):
    def __init__(self, horizon, ch_dim=2, num_dims=5):
        """
        horizon: 表示要添加的时间步数。
        ch_dim: 表示通道维度的索引, 默认为2, 表示在通道维度上添加时间条件信息。
        num_dims: 表示输入张量的维度, 默认为5, 表示输入是一个5维张量(通常用于视频数据），如果是其他维度数，则会进行相应的调整。
        """
        super().__init__()
        
        # 该函数在被调用时，horizon 的值由 forecast_steps 赋予
        self.horizon = horizon
        self.ch_dim = ch_dim
        self.num_dims = num_dims

    def forward(self, x, fstep=0):
        """
        x: 输入图像或张量。
        fstep: 表示要添加的时间步(默认为0)
        """
        
        # 检查输入张量的维度数是否为5
        if self.num_dims == 5:
            bs, seq_len, ch, h, w = x.shape
            ct = condition_time(x, fstep, (h, w), seq_len=self.horizon).repeat(bs, seq_len, 1, 1, 1)
        else:
            bs, h, w, ch = x.shape
            ct = condition_time(x, fstep, (h, w), seq_len=self.horizon).repeat(bs, 1, 1, 1)
            ct = ct.permute(0, 2, 3, 1)
        
        # 将输入张量和时间条件热图在指定的通道维度上拼接，以添加时间条件信息。
        x = torch.cat([x, ct], dim=self.ch_dim)
        
        # 断言，确保拼接后的张量通道数等于输入通道数加上时间步数，以验证操作的正确性。
        assert x.shape[self.ch_dim] == (ch + self.horizon)  # check if it makes sense
        return x
