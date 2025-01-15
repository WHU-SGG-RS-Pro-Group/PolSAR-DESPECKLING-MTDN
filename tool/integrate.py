import torch
import torch.nn as nn
import torch.nn.functional as F
from mpmath import sigmoid

from model import Fea_Tfusion,Fea_extract
from torch.optim import Adam, lr_scheduler,SGD
# from models.networks import BASE_Transformer, get_scheduler
from models.losses import cross_entropy
from rdn import RDB
from attention import RCSA
from Cross_A import CroA
import functools
from einops import rearrange

import models
from models.help_funcs import Transformer, TransformerDecoder, TwoLayerConv2d
class ResNet(torch.nn.Module):
    def __init__(self, input_nc, output_nc,
                 resnet_stages_num=5, backbone='resnet18',
                 output_sigmoid=False, if_upsample_2x=True):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(ResNet, self).__init__()
        expand = 1
        if backbone == 'resnet18':
            self.resnet = models.resnet18(pretrained=False,
                                          replace_stride_with_dilation=[False,True,True])
            # self.resnet = ModifiedResNet18(in_channels=4)
        elif backbone == 'resnet34':
            self.resnet = models.resnet34(pretrained=True,
                                          replace_stride_with_dilation=[False,True,True])
        elif backbone == 'resnet50':
            self.resnet = models.resnet50(pretrained=True,
                                          replace_stride_with_dilation=[False,True,True])
            expand = 4
        else:
            raise NotImplementedError
        self.relu = nn.ReLU()
        self.upsamplex2 = nn.Upsample(scale_factor=2)
        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')

        self.classifier = TwoLayerConv2d(in_channels=32, out_channels=output_nc)

        self.resnet_stages_num = resnet_stages_num

        self.if_upsample_2x = if_upsample_2x
        if self.resnet_stages_num == 5:
            layers = 512 * expand
        elif self.resnet_stages_num == 4:
            layers = 256 * expand
        elif self.resnet_stages_num == 3:
            layers = 128 * expand
        else:
            raise NotImplementedError
        self.conv_pred = nn.Conv2d(layers, 32, kernel_size=3, padding=1)

        self.output_sigmoid = output_sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        x1 = self.forward_single(x1)
        x2 = self.forward_single(x2)
        x = torch.abs(x1 - x2)
        if not self.if_upsample_2x:
            x = self.upsamplex2(x)
        x = self.upsamplex4(x)
        x = self.classifier(x)

        if self.output_sigmoid:
            x = self.sigmoid(x)
        return x

    def forward_single(self, x):
        # resnet layers
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x_4 = self.resnet.layer1(x) # 1/4, in=64, out=64
        x_8 = self.resnet.layer2(x_4) # 1/8, in=64, out=128

        if self.resnet_stages_num > 3:
            x_8 = self.resnet.layer3(x_8) # 1/8, in=128, out=256

        if self.resnet_stages_num == 5:
            x_8 = self.resnet.layer4(x_8) # 1/32, in=256, out=512
        elif self.resnet_stages_num > 5:
            raise NotImplementedError

        if self.if_upsample_2x:
            x = self.upsamplex2(x_8)
        else:
            x = x_8
        # output layers
        x = self.conv_pred(x)
        return x

class integrate(ResNet):
    def __init__(self, input_nc, output_nc, with_pos, resnet_stages_num=5,
                token_len=4, token_trans=True,
                enc_depth=1, dec_depth=1,
                dim_head=64, decoder_dim_head=64,
                tokenizer=True, if_upsample_2x=True,
                pool_mode='max', pool_size=2,
                backbone='resnet18',
                decoder_softmax=True, with_decoder_pos=None,
                with_decoder=True,args=False):
        super(integrate, self).__init__(input_nc, output_nc, backbone=backbone,
                                                   resnet_stages_num=resnet_stages_num,
                                                   if_upsample_2x=if_upsample_2x,
                                                   )
        #denoised
        ncha = args.ncha
        nDense = args.nDense
        nFeat = args.nFeat
        growthRate = args.growthRate
        self.args = args

        # F-1
        self.conv1 = nn.Conv2d(64, nFeat, kernel_size=3, padding=1, bias=True)
        self.conv1.name = 'denoised_conv1'
        # F0
        self.conv2 = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True)
        self.conv2.name = 'denoised_conv2'
        # RDBs 3
        self.RDB1 = RDB(nFeat, nDense, growthRate)
        self.RDB1.name = 'denoised_RDB1'
        self.RDB2 = RDB(nFeat, nDense, growthRate)
        self.RDB2.name = 'denoised_RDB2'
        self.RDB3 = RDB(nFeat, nDense, growthRate)
        self.RDB3.name = 'denoised_RDB3'


        # global feature fusion (GFF)
        self.GFF_1x1 = nn.Conv2d(nFeat * 3, nFeat, kernel_size=1, padding=0, bias=True)
        self.GFF_1x1.name = 'denoised_GFF_1x1'
        self.GFF_3x3 = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True)
        self.GFF_3x3.name = 'denoised_GFF_3x3'
        # conv
        self.conv3 = nn.Conv2d(nFeat, ncha, kernel_size=3, padding=1, bias=True)
        self.conv3.name = 'denoised_conv3'

        # attention
        self.rcsa1 = RCSA(nFeat=64)
        self.rcsa1.name = 'cd_rcsa1'
        self.rcsa2 = RCSA(nFeat=64)
        self.rcsa2.name = 'cd_rcsa2'
        self.rcsa3 = RCSA(nFeat=64)
        self.rcsa3.name = 'cd_rcsa3'
        self.croa1 = CroA(64)
        self.croa1.name = 'cd_croa1'
        self.TFusion = Fea_Tfusion()
        self.TFusion.name = 'cd_TFusion'
        self.Fea_extra = Fea_extract(ncha, 64)
        self.Fea_extra.name = 'cd_Fea_extra'

        #cd
        self.token_len = token_len
        self.conv_a = nn.Conv2d(32, self.token_len, kernel_size=1,
                                padding=0, bias=False)
        self.conv_a.name='cd_conv_a'
        self.tokenizer = tokenizer
        if not self.tokenizer:
            #  if not use tokenzier，then downsample the feature map into a certain size
            self.pooling_size = pool_size
            self.pool_mode = pool_mode
            self.token_len = self.pooling_size * self.pooling_size

        self.token_trans = token_trans
        self.with_decoder = with_decoder
        dim = 32
        mlp_dim = 2 * dim

        self.with_pos = with_pos
        if with_pos is 'learned':
            self.pos_embedding = nn.Parameter(torch.randn(1, self.token_len * 2, 32))
            self.pos_embedding.name = 'cd_pos_embedding'
        decoder_pos_size = 256 // 4
        self.with_decoder_pos = with_decoder_pos
        if self.with_decoder_pos == 'learned':
            self.pos_embedding_decoder = nn.Parameter(torch.randn(1, 32,
                                                                  decoder_pos_size,
                                                                  decoder_pos_size))
            self.pos_embedding_decoder.name = 'cd_pos_embedding'
        self.enc_depth = enc_depth
        self.dec_depth = dec_depth
        self.dim_head = dim_head
        self.decoder_dim_head = decoder_dim_head
        self.transformer = Transformer(dim=dim, depth=self.enc_depth, heads=8,
                                       dim_head=self.dim_head,
                                       mlp_dim=mlp_dim, dropout=0)
        self.transformer.name='cd_transformer'
        self.transformer_decoder = TransformerDecoder(dim=dim, depth=self.dec_depth,
                                                      heads=8, dim_head=self.decoder_dim_head, mlp_dim=mlp_dim,
                                                      dropout=0,
                                                      softmax=decoder_softmax)
        self.transformer_decoder.name = 'cd_transformer_decoder'

    def forward(self, T1,T2):
        # net1——change detect
        x1 = T1.narrow(1, 0, 4)
        x2 = T2.narrow(1, 0, 4)
        # forward backbone resnet
        x1 = self.forward_single(x1)
        x2 = self.forward_single(x2)

        #  forward tokenzier
        if self.tokenizer:
            token1 = self._forward_semantic_tokens(x1)
            token2 = self._forward_semantic_tokens(x2)
        else:
            token1 = self._forward_reshape_tokens(x1)
            token2 = self._forward_reshape_tokens(x2)
        # forward transformer encoder
        if self.token_trans:
            self.tokens_ = torch.cat([token1, token2], dim=1)
            self.tokens = self._forward_transformer(self.tokens_)
            token1, token2 = self.tokens.chunk(2, dim=1)
        # forward transformer decoder
        if self.with_decoder:
            x1 = self._forward_transformer_decoder(x1, token1)
            x2 = self._forward_transformer_decoder(x2, token2)
        else:
            x1 = self._forward_simple_decoder(x1, token1)
            x2 = self._forward_simple_decoder(x2, token2)

        # feature differencing
        x_cd = torch.abs(x1 - x2)
        if not self.if_upsample_2x:
            x_cd = self.upsamplex2(x_cd)
        x_cd = self.upsamplex4(x_cd)
        # forward small cnn
        x_CD = self.classifier(x_cd)
        if self.output_sigmoid:
            x_CD = self.sigmoid(x_CD)

        # net2--denoised
        x_1=torch.mul(T1,F.sigmoid(x_cd))
        x_2=torch.mul(T2,F.sigmoid(x_cd))
        x_1=self.Fea_extra(x_1).unsqueeze(1)
        x_2=self.Fea_extra(x_2).unsqueeze(1)
        fea = torch.cat(x_1,x_2, dim=1)  # B, N, C, H, W
        fea = self.TFusion(fea)

        # F_ = self.conv1(x)
        F_ = self.conv1(fea)
        F_0 = self.conv2(F_)
        F_1 = self.rcsa1(self.RDB1(F_0))
        F_2 = self.rcsa2(self.RDB2(F_1))
        F_3 = self.rcsa3(self.RDB3(F_2))
        FF = torch.cat((F_1, F_2, F_3), 1)
        FdLF = self.GFF_1x1(FF)
        FGF = self.GFF_3x3(FdLF)
        FDF = FGF + F_
        x_DE = self.conv3(self.croa1(FDF, F_))
        return x_CD,x_DE

