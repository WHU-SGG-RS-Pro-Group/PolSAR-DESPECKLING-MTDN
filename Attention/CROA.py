import torch.nn as nn
import torch
class CroA(nn.Module):
    """ Cross Attention
    """
    def __init__(self, nFeat):
        super(CroA, self).__init__()
        self.croa_conv1 = nn.Sequential(
            nn.Conv2d(nFeat, nFeat, kernel_size=3,  stride=1, padding=1, bias=True), nn.PReLU())
        self.croa_conv2 = nn.Sequential(
            nn.Conv2d(nFeat, nFeat, kernel_size=3,  stride=1, padding=1, bias=True), nn.PReLU())
        self.croa_conv3 = nn.Sequential(
            nn.Conv2d(nFeat, nFeat, kernel_size=3,  stride=1, padding=1, bias=True), nn.Sigmoid())
        self.croa_conv4 = nn.Sequential(
            nn.Conv2d(nFeat, nFeat, kernel_size=3,  stride=1, padding=1, bias=True), nn.PReLU())
        self.croa_conv1x1_1 = nn.Sequential(
            nn.Conv2d(nFeat, 1, kernel_size=1, padding=0, bias=True), nn.Sigmoid())

        self.croa_conv1x1_2 = nn.Sequential(
            nn.Conv2d(nFeat*2, nFeat, kernel_size=1, padding=0, bias=True), nn.PReLU())

    def forward(self, lr, hr):
        lr_fea = self.croa_conv1(lr)
        hr_fea = self.croa_conv2(hr)
        a=torch.cat((torch.mul(lr_fea, self.croa_conv1x1_1(hr_fea)), torch.mul(hr_fea, self.croa_conv3(lr_fea))), 1)
        out = self.croa_conv4(self.croa_conv1x1_2(torch.cat((torch.mul(lr_fea, self.croa_conv1x1_1(hr_fea)), torch.mul(hr_fea, self.croa_conv3(lr_fea))), 1)))
        return out
