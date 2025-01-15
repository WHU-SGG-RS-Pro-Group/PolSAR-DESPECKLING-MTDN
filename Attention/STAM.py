import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F




def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)

class ResidualBlock_noBN(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # initialization
        self._initialize_weights()

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out

    def _initialize_weights(self) :
        for m in self.modules() :
            if isinstance(m, nn.Conv2d) :
                init.orthogonal_(m.weight)
                print('init weight')
                if m.bias is not None :
                    init.constant_(m.bias, 0)
def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)

class Fea_extract(nn.Module):

    def __init__(self,inchannel=1,outchannel=64):
        super(Fea_extract,self).__init__()
        self.fea = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 3, 1, 1),
            nn.ReLU()
        )
        layers1 = []
        for i in range(5):
            layers1.append(ResidualBlock_noBN(outchannel))

        self.fea_extraction = nn.Sequential(*layers1)

    def forward(self,x):
        x1=self.fea(x)
        out=self.fea_extraction(x1)
        return out

class TSA_Fusion(nn.Module):
    ''' Temporal Spatial Attention fusion module
    Temporal: correlation;
    Spatial: 3 pyramid levels.
    '''

    def __init__(self, outchannels, ntemporal, center):
        super(TSA_Fusion, self).__init__()
        self.center = center
        # temporal attention (before fusion conv)
        self.tAtt_1 = nn.Conv2d(outchannels, outchannels, 3, 1, 1, bias=True)
        self.tAtt_2 = nn.Conv2d(outchannels, outchannels, 3, 1, 1, bias=True)

        # fusion conv: using 1x1 to save parameters and computation
        self.fea_fusion = nn.Conv2d(ntemporal * outchannels, outchannels, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, extraction_fea):
        B, N, C, H, W = extraction_fea.size()  # N video frames
        #### temporal attention
        emb_ref = self.tAtt_2(extraction_fea[:, self.center, :, :, :].clone())
        emb = self.tAtt_1(extraction_fea.view(-1, C, H, W)).view(B, N, -1, H, W)  # [B, N, C(nf), H, W]

        cor_l = []
        for i in range(N):
            emb_nbr = emb[:, i, :, :, :]
            cor_tmp = torch.mean(emb_nbr * emb_ref, 1).unsqueeze(1)  # B, 1, H, W
            cor_l.append(cor_tmp)
        cor_prob = torch.sigmoid(torch.cat(cor_l, dim=1) ) # B, N, H, W
        cor_prob = cor_prob.unsqueeze(2).repeat(1, 1, C, 1, 1).view(B, -1, H, W)
        extraction_fea = extraction_fea.view(B, -1, H, W) * cor_prob

        #### fusion
        fea = self.lrelu(self.fea_fusion(extraction_fea))
        return fea

class Fea_Tfusion_diff(nn.Module):

    def __init__(self,outchannel=64):
        super(Fea_Tfusion_diff,self).__init__()
        self.tAtt_1 = nn.Conv2d(outchannel, outchannel, 3, 1, 1, bias=True)
        self.tAtt_2 = nn.Conv2d(outchannel, outchannel, 3, 1, 1, bias=True)


        # spatial attention (after fusion conv)
        self.sAtt_1 = nn.Conv2d( outchannel, outchannel, 1, 1, bias=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.avgpool = nn.AvgPool2d(3, stride=2, padding=1)
        self.sAtt_2 = nn.Conv2d(outchannel * 2, outchannel, 1, 1, bias=True)
        self.sAtt_3 = nn.Conv2d(outchannel, outchannel, 3, 1, 1, bias=True)
        self.sAtt_4 = nn.Conv2d(outchannel, outchannel, 1, 1, bias=True)
        self.sAtt_5 = nn.Conv2d(outchannel, outchannel, 3, 1, 1, bias=True)
        self.sAtt_L1 = nn.Conv2d(outchannel, outchannel, 1, 1, bias=True)
        self.sAtt_L2 = nn.Conv2d(outchannel * 2, outchannel, 3, 1, 1, bias=True)
        self.sAtt_L3 = nn.Conv2d(outchannel, outchannel, 3, 1, 1, bias=True)
        self.sAtt_add_1 = nn.Conv2d(outchannel, outchannel, 1, 1, bias=True)
        self.sAtt_add_2 = nn.Conv2d(outchannel, outchannel, 1, 1, bias=True)


        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.fea_fusion = nn.Conv2d(outchannel*2, outchannel, 1, 1, bias=True)

    def forward(self, Fea_ext):
        B, N, C, H, W = Fea_ext.size()  # N video frames
        #### temporal attention
        emb_ref=self.tAtt_1(Fea_ext[:,0,:,:,:])


        cor_l = []
        for i in range(N):
            emb_nbr = self.tAtt_2(Fea_ext[ :,i, :, :, :])
            cor_tmp = torch.exp(-1*torch.abs(torch.mean(emb_nbr - emb_ref, 1).unsqueeze(1)))  # B, 1, H, W
            cor_l.append(cor_tmp)
        cor_prob = torch.sigmoid(torch.cat(cor_l, dim=1))  # B, N, H, W
        cor_prob = cor_prob.unsqueeze(2).repeat(1, 1, C, 1, 1)
        extraction_fea = torch.mean(Fea_ext* cor_prob,dim=1)
        #### fusion
        # fea = self.lrelu(self.fea_fusion(extraction_fea))



        #### spatial attention
        fea=Fea_ext[:,0,:,:,:]
        att = self.lrelu(self.sAtt_1(fea))
        att_max = self.maxpool(att)
        att_avg = self.avgpool(att)
        att = self.lrelu(self.sAtt_2(torch.cat([att_max, att_avg], dim=1)))
        # att = self.lrelu(self.sAtt_2(att_avg))
        # pyramid levels
        att_L = self.lrelu(self.sAtt_L1(att))
        att_max = self.maxpool(att_L)
        att_avg = self.avgpool(att_L)
        att_L = self.lrelu(self.sAtt_L2(torch.cat([att_max, att_avg], dim=1)))
        # att_L = self.lrelu(self.sAtt_L2( att_avg))
        att_L = self.lrelu(self.sAtt_L3(att_L))
        att_L = F.interpolate(att_L, scale_factor=2, mode='bilinear', align_corners=False)
        # att_L=self.sAtt_up_1(att_L)

        att = self.lrelu(self.sAtt_3(att))
        att = att + att_L
        att = self.lrelu(self.sAtt_4(att))
        # att = self.sAtt_up_2(att)
        att = F.interpolate(att, scale_factor=2, mode='bilinear', align_corners=False)
        att = self.sAtt_5(att)
        att_add = self.sAtt_add_2(self.lrelu(self.sAtt_add_1(att)))
        att = torch.sigmoid(att)

        fea = fea * att * 2 + att_add

        #### fusion
        fea = self.lrelu(self.fea_fusion(torch.cat((extraction_fea,fea),dim=1)))


        return fea

class Fea_Tfusion(nn.Module):#

    def __init__(self,outchannel=64):
        super(Fea_Tfusion,self).__init__()
        self.tAtt_1 = nn.Conv2d(outchannel, outchannel, 3, 1, 1, bias=True)
        self.tAtt_2 = nn.Conv2d(outchannel, outchannel, 3, 1, 1, bias=True)


        # spatial attention (after fusion conv)
        self.sAtt_1 = nn.Conv2d( outchannel, outchannel, 1, 1, bias=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.avgpool = nn.AvgPool2d(3, stride=2, padding=1)
        self.sAtt_2 = nn.Conv2d(outchannel * 2, outchannel, 1, 1, bias=True)
        self.sAtt_3 = nn.Conv2d(outchannel, outchannel, 3, 1, 1, bias=True)
        self.sAtt_4 = nn.Conv2d(outchannel, outchannel, 1, 1, bias=True)
        self.sAtt_5 = nn.Conv2d(outchannel, outchannel, 3, 1, 1, bias=True)
        self.sAtt_L1 = nn.Conv2d(outchannel, outchannel, 1, 1, bias=True)
        self.sAtt_L2 = nn.Conv2d(outchannel * 2, outchannel, 3, 1, 1, bias=True)
        self.sAtt_L3 = nn.Conv2d(outchannel, outchannel, 3, 1, 1, bias=True)
        self.sAtt_add_1 = nn.Conv2d(outchannel, outchannel, 1, 1, bias=True)
        self.sAtt_add_2 = nn.Conv2d(outchannel, outchannel, 1, 1, bias=True)


        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.fea_fusion = nn.Conv2d(outchannel*2, outchannel, 1, 1, bias=True)

    def forward(self, Fea_ext):
        B, N, C, H, W = Fea_ext.size()  # N video frames
        #### temporal attention
        emb_ref=self.tAtt_1(Fea_ext[:,0,:,:,:])


        cor_l = []
        for i in range(N):
            emb_nbr = self.tAtt_2(Fea_ext[ :,i, :, :, :])
            cor_tmp = torch.mean(emb_nbr * emb_ref, 1).unsqueeze(1)  # B, 1, H, W
            cor_l.append(cor_tmp)
        cor_prob = torch.sigmoid(torch.cat(cor_l, dim=1))  # B, N, H, W
        cor_prob = cor_prob.unsqueeze(2).repeat(1, 1, C, 1, 1)
        extraction_fea = torch.mean(Fea_ext* cor_prob,dim=1)

        #### spatial attention
        fea=Fea_ext[:,0,:,:,:]
        att = self.lrelu(self.sAtt_1(fea))
        att_max = self.maxpool(att)
        att_avg = self.avgpool(att)
        att = self.lrelu(self.sAtt_2(torch.cat([att_max, att_avg], dim=1)))
        # att = self.lrelu(self.sAtt_2(att_avg))
        # pyramid levels
        att_L = self.lrelu(self.sAtt_L1(att))
        att_max = self.maxpool(att_L)
        att_avg = self.avgpool(att_L)
        att_L = self.lrelu(self.sAtt_L2(torch.cat([att_max, att_avg], dim=1)))
        # att_L = self.lrelu(self.sAtt_L2( att_avg))
        att_L = self.lrelu(self.sAtt_L3(att_L))
        att_L = F.interpolate(att_L, scale_factor=2, mode='bilinear', align_corners=False)
        # att_L=self.sAtt_up_1(att_L)

        att = self.lrelu(self.sAtt_3(att))
        att = att + att_L
        att = self.lrelu(self.sAtt_4(att))
        # att = self.sAtt_up_2(att)
        att = F.interpolate(att, scale_factor=2, mode='bilinear', align_corners=False)
        att = self.sAtt_5(att)
        att_add = self.sAtt_add_2(self.lrelu(self.sAtt_add_1(att)))
        att = torch.sigmoid(att)

        fea = fea * att * 2 + att_add

        #### fusion
        fea = self.lrelu(self.fea_fusion(torch.cat((extraction_fea,fea),dim=1)))


        return fea


class Fea_Tfusion_(nn.Module):

    def __init__(self,outchannel=64):
        super(Fea_Tfusion_,self).__init__()
        self.tAtt_1 = nn.Conv2d(outchannel, outchannel, 3, 1, 1, bias=True)
        self.tAtt_2 = nn.Conv2d(outchannel, outchannel, 3, 1, 1, bias=True)
        self.fea_fusion = nn.Conv2d(outchannel, outchannel, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, Fea_ext):
        B, N, C, H, W = Fea_ext.size()  # N video frames
        #### temporal attention
        emb_ref=self.tAtt_1(Fea_ext[:,0,:,:,:])


        cor_l = []
        for i in range(N):
            emb_nbr = self.tAtt_2(Fea_ext[ :,i, :, :, :])
            cor_tmp = (emb_nbr * emb_ref).unsqueeze(1)  # B, 1, C,H, W
            cor_l.append(cor_tmp)
        cor_prob = torch.sigmoid(torch.cat(cor_l, dim=1))  # B, N,C, H, W
        # cor_prob = cor_prob.unsqueeze(2).repeat(1, 1, C, 1, 1)
        extraction_fea = torch.mean(Fea_ext* cor_prob,dim=1).contiguous()

        #### fusion
        fea = self.lrelu(self.fea_fusion(extraction_fea))
        return fea

class Fea_recon(nn.Module):

    def __init__(self,inchannel=1,outchannel=64):
        super(Fea_recon,self).__init__()
        self.fea_extr=Fea_extract(inchannel,outchannel)
        self.fea_fusion=Fea_Tfusion(outchannel)
        layers2 = []
        for i in range(10):
            layers2.append(ResidualBlock_noBN(outchannel))

        self.fea_reconstruction = nn.Sequential(*layers2)
        self.conv_last = nn.Conv2d(outchannel, inchannel, 3, 1, 1, bias=True)

    def forward(self,x):
        # print(x.size())
        B, N, C, H, W=x.size()
        fea = []
        for i in range(N):
            x_nbr = x[:, i, :, :, :].contiguous()
            fea_tmp = self.fea_extr(x_nbr).unsqueeze(1)  # B, 1, C, H, W
            fea.append(fea_tmp)
        fea = torch.cat(fea, dim=1)  # B, N, C, H, W
        fea=self.fea_fusion(fea)
        res = self.fea_reconstruction(fea)
        out = self.conv_last(res)
        out = x[:,0,:,:,:].contiguous() + out
        return out

class Fea_recon_NoTSA(nn.Module):

    def __init__(self,inchannel=1,outchannel=64):
        super(Fea_recon_NoTSA,self).__init__()
        self.fea_extr=Fea_extract(inchannel,outchannel)
        # self.fea_fusion=Fea_Tfusion(outchannel)
        layers2 = []
        for i in range(10):
            layers2.append(ResidualBlock_noBN(outchannel))

        self.fea_reconstruction = nn.Sequential(*layers2)
        self.conv_last = nn.Conv2d(outchannel, inchannel, 3, 1, 1, bias=True)

    def forward(self,x):
        # print(x.size())
        B, N, C, H, W=x.size()
        fea = []
        for i in range(N):
            x_nbr = x[:, i, :, :, :].contiguous()
            fea_tmp = self.fea_extr(x_nbr).unsqueeze(1)  # B, 1, C, H, W
            fea.append(fea_tmp)
        fea = torch.cat(fea, dim=1)  # B, N, C, H, W
        # fea=self.fea_fusion(fea)
        fea=torch.mean(fea,dim=1).contiguous()
        res = self.fea_reconstruction(fea)
        out = self.conv_last(res)
        out = x[:,0,:,:,:].contiguous() + out
        return out

class TSFea_extrac(nn.Module):

    def __init__(self,inchannel=1, outchannel= 64,ntemporal=5,center=None):
        super(TSFea_extrac,self).__init__()
        self.center = ntemporal//2 if center is None else center
        # self.feature_extraction=feature_etraction(inchannel,outchannel,ntemporal,self.center)
        self.fea = nn.Sequential(
            nn.Conv2d(1, outchannel//2, 3, 1, 1),
            nn.ReLU()
        )
        layers1=[]
        for i in range(5):
            layers1.append(ResidualBlock_noBN(outchannel//2))

        self.fea_extraction=nn.Sequential(*layers1)
        self.TSA=TSA_Fusion(outchannel//2,ntemporal,self.center)
        layers2 = []
        for i in range(10):
            layers2.append(ResidualBlock_noBN(outchannel))

        self.fea_reconstruction = nn.Sequential(*layers2)
        self.conv_last=nn.Conv2d(outchannel, 1, 3, 1, 1, bias=True)

    def forward(self,x):
        B, N, C, H, W = x.size()  # N :n temporal
        x_center = x[:, self.center, :, :, :].contiguous()
        # x_mean = torch.mean(x,dim=1)
        x = x.view(-1, C, H, W)
        fea_first=self.fea(x)
        extrac_fea=self.fea_extraction(fea_first)
        extrac_fea=extrac_fea.view(B,N,-1,H,W)
        fea_center=extrac_fea[:, self.center, :, :, :].view(B,-1,H,W)
        TSA_fea=self.TSA(extrac_fea)

        fea=torch.cat((fea_center,TSA_fea),dim=1)
        res=self.fea_reconstruction(fea)
        out=self.conv_last(res)
        out=x_center+out
        # out=torch.div(self.center,out)
        return out






