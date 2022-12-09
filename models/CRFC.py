# -*- coding: utf-8 -*-
# @Time    : 2022/11/10
# @Author  : White Jiang
import torch
import torch.nn as nn

class CRA_DOT(nn.Module):
    def __init__(self):
        super(CRA_DOT, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, y):
        """
        :param x: 下一阶段特征
        :param y: 前一阶段特征
        :return:
        """
        x_y = torch.mul(x, y)
        b, c, h, w = x.size()
        x_y = x_y.reshape(b, c, h * w)
        # print(x_y)
        x_scale = self.softmax(x_y)
        # print(x_scale)
        x_scale = x_scale.reshape(b, c, h, w)
        y_y = torch.mul(y, y)
        y_y = y_y.reshape(b, c, h * w)
        y_y = y_y - x_y
        # print(y_y)
        y_scale = self.softmax(y_y)
        # print(y_scale)
        y_scale = y_scale.reshape(b, c, h, w)
        out_x = x_scale * x
        out_y = y_scale * y
        return out_x, out_y


if __name__ == '__main__':
    a = torch.rand((1, 2, 4, 4)).cuda()
    scale_max = torch.max(a, dim=-1, keepdim=True)[0]
    # scale = scale.view(b, c, h, w)
    b = torch.clamp((a < scale_max * 0.95).float(), min=0.0)
    # c = torch.rand((2, 1024))
    # print((c[:, :512]).size())
    se = CRA_DOT().cuda()
    se(a, b)
    # print(a.size())
    # print(a)
    # _, ids = torch.sort(a, -1, descending=True)
    # print(_)
    # print(ids)
    # print(ids[:, :, :2])
    # se = CRFC()
    # se(a, b)
