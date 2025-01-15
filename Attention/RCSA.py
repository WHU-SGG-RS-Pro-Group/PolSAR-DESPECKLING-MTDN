
# python3
# -*- coding: utf-8 -*-
# @Time    : 2022/3/3 9:24
# @Author  : Liupeng Lin
# @Email   : linliupeng@whu.edu.cn
# Copyright (C) 2022 Liupeng Lin. All Rights Reserved.


import torch
from torch import nn

#channel attention
class CAL(nn.Module):
    def __init__(self, nFeat, ratio=16):
        super(CAL, self).__init__()
        self.cal_conv1 = nn.Sequential(
            nn.Conv2d(nFeat, nFeat, kernel_size=3,  stride=1, padding=1, bias=True), nn.PReLU())
        self.cal_conv2 = nn.Sequential(
            nn.Conv2d(nFeat, nFeat, kernel_size=1, padding=0, bias=True), nn.PReLU())
        self.cal_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.cal_fc1 = nn.Sequential(
            nn.Conv2d(nFeat, nFeat // ratio, 1, padding=0, bias=True), nn.PReLU())
        self.cal_fc2 = nn.Sequential(
            nn.Conv2d(nFeat // ratio, nFeat, 1, padding=0, bias=True), nn.Sigmoid())

    def forward(self, x):
        cal_weight_avg = self.cal_fc2(self.cal_fc1(self.cal_avg_pool(x)))
        out = self.cal_conv2(torch.mul(x, cal_weight_avg))
        return out

#spatial attention
class SAL(nn.Module):
    def __init__(self, nFeat):
        super(SAL, self).__init__()
        self.sal_conv1 = nn.Sequential(
            nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, groups=nFeat, bias=True), nn.PReLU())
        self.sal_conv1x1 = nn.Sequential(
            nn.Conv2d(nFeat, 1, kernel_size=1, padding=0, bias=True), nn.Sigmoid())
        self.sal_conv2 = nn.Sequential(
            nn.Conv2d(nFeat, nFeat, kernel_size=3, stride=1, padding=1, bias=True), nn.PReLU())

    def forward(self, x):
        sal_weight = self.sal_conv1x1(self.sal_conv1(x))
        out = self.sal_conv2(x * sal_weight)
        return out


class RCSA(nn.Module):
    def __init__(self, nFeat):
        super(RCSA, self).__init__()
        self.rcsa_cal = CAL(nFeat)
        self.rcsa_sal = SAL(nFeat)
        self.rcsa_conv1 = nn.Sequential(
            nn.Conv2d(nFeat, nFeat, kernel_size=3, stride=1, padding=1, bias=True), nn.PReLU())

    def forward(self, x):
        out1 = self.rcsa_cal(x)
        out2 = self.rcsa_sal(x)
        out = torch.add(x, self.rcsa_conv1(torch.add(out1, out2)))
        return out