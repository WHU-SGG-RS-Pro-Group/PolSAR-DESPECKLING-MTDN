# python3
# -*- coding: utf-8 -*-
# @Time    : 2021/11/24 20:42
# @Author  : Liupeng Lin
# @Email   : linliupeng@whu.edu.cn
# Copyright (C) 2021 Liupeng Lin. All Rights Reserved.


import torch.nn.functional as F
from Attention.RCSA import RCSA
from Attention.CROA import CroA
from Attention.STAM import Fea_Tfusion,Fea_extract
from Attention.GCA import *

class make_dense(nn.Module):
    def __init__(self, nChannels, growthRate, kernel_size=3):
        super(make_dense, self).__init__()
        self.conv = nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=(kernel_size-1)//2, bias=False)



    def forward(self, x):
        x1=self.conv(x)
        # x_bn=self.bn1(x1)   #BN
        out = F.relu(x1)
        out = torch.cat((x, out), 1)
        return out


# Residual dense block (RDB) architecture
class RDB(nn.Module):
    def __init__(self, nFeat, nDense, growthRate):
        super(RDB, self).__init__()
        nFeat_ = nFeat
        modules = []
        for i in range(nDense):
            modules.append(make_dense(nFeat_, growthRate))
            nFeat_ += growthRate
        self.dense_layers = nn.Sequential(*modules)
        self.conv_1x1 = nn.Conv2d(nFeat_, nFeat, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        out = self.dense_layers(x)
        out = self.conv_1x1(out)
        out = out + x
        return out


# Residual Dense Network
class DSDN(nn.Module):
    def __init__(self, args):
        super(DSDN, self).__init__()
        # ncha_clm = int(args.ncha_clm/2)
        # ncha_modis = args.ncha_modis
        ncha = args.ncha
        nDense = args.nDense
        nFeat = args.nFeat
        growthRate = args.growthRate
        self.args = args

        # self.GCA=LSKblock(64)

        self.GCA1=LSKblock(64)
        self.GCA2=LSKblock(64)
        self.GCA3 = LSKblock(64)
        self.GCA4 = LSKblock(64)
        self.GCA5 = LSKblock(64)
        self.GCA6 = LSKblock(64)
        self.GCA7 = LSKblock(64)
        self.GCA8 = LSKblock(64)
        # F-1
        self.conv1 = nn.Conv2d(64, nFeat, kernel_size=3, padding=1, bias=True)

        # F0
        self.conv2 = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True)
        self.conv_1x1= nn.Conv2d(32 , nFeat, kernel_size=1, padding=0, bias=True)

        # RDBs 3
        self.RDB1 = RDB(nFeat, nDense, growthRate)
        self.RDB2 = RDB(nFeat, nDense, growthRate)
        self.RDB3 = RDB(nFeat, nDense, growthRate)

        # global feature fusion (GFF)
        self.GFF_1x1 = nn.Conv2d(nFeat*3, nFeat, kernel_size=1, padding=0, bias=True)
        self.GFF_3x3 = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True)

        # conv
        self.conv3 = nn.Conv2d(nFeat, ncha, kernel_size=3, padding=1, bias=True)

        #attention
        self.rcsa1 = RCSA(nFeat=64)
        self.rcsa2 = RCSA(nFeat=64)
        self.rcsa3 = RCSA(nFeat=64)
        self.croa1 = CroA(64)


        self.TFusion = Fea_Tfusion()
        self.Fea_extra = Fea_extract(ncha, 64)


    def forward(self, x,x_mid):

        fea = []
        for i in range(2):
            # x_nbr = x.narrow(1, i * int(self.args.ncha_modis), int(self.args.ncha_modis)).contiguous()
            x_nbr = x.narrow(1, i * int(self.args.ncha), int(self.args.ncha)).contiguous()
            fea_tmp = self.Fea_extra(x_nbr).unsqueeze(1)  # B, 1, C, H, W
            fea.append(fea_tmp)
        fea = torch.cat(fea, dim=1)  # B, N, C, H, W
        fea = self.TFusion(fea)


        F_ = self.conv1(fea)
        F_0 = self.conv2(F_)
        F_0_1 = self.GCA5(self.GCA4(self.GCA3(self.GCA2(self.GCA1(F_0)))))
        x_mid = torch.sigmoid(self.conv_1x1(x_mid))
        F_0_2 = F_0_1 + torch.mul(F_0_1, x_mid)
        F_1 =self.GCA6(self.rcsa1(self.RDB1(F_0_2)))
        F_2 = self.GCA7(self.rcsa2(self.RDB2(F_1)))
        F_3 = self.GCA8(self.rcsa3(self.RDB3(F_2)))
        FF = torch.cat((F_1, F_2, F_3), 1)
        FdLF = self.GFF_1x1(FF)
        FGF = self.GFF_3x3(FdLF)
        FDF = FGF + F_
        output = self.conv3(self.croa1(FDF, F_))
        return output