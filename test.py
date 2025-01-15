#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tool.datasets import load_dataset
from noise2noise import Noise2Noise

from argparse import ArgumentParser


def parse_args():
    """Command-line argument parser for testing."""

    # New parser
    parser = ArgumentParser(description='PyTorch implementation of Noise2Noise from Lehtinen et al. (2018)')

    # Data parametersv
    parser.add_argument('-d', '--data', help='dataset root path', default=r'./../MTDN/data/test')
    parser.add_argument('--load-ckpt', help='load model checkpoint',default=r'./../MTDN/models/denoised/n2n-epoch100.pth')
    parser.add_argument('--show-output', help='pop up window to display outputsi', default=40, type=int)
    parser.add_argument('--cuda', help='use cuda', default=True,type=bool)
    parser.add_argument('--batch_size', default=4, type=int, help='batch size')
    # parser.add_argument('--bos', default=10, type=int, help='border_size')
    # parser.add_argument('--epoch', default=200, type=int, help='number of train epoch')
    # Pre_model
    parser.add_argument('--trained_model_ckpt', help='training baed on Pre trained model ', default='', type=str)
    # Pre_Optim
    parser.add_argument('--pre_Adam', help='Pre Adam parameters', default='', type=str)
    parser.add_argument('--nDense', type=int, default=6, help='nDenselayer of RDB')
    parser.add_argument('--growthRate', type=int, default=32, help='growthRate of dense net')
    parser.add_argument('--nFeat', type=int, default=64, help='number of feature maps')
    parser.add_argument('--ncha_modis', type=int, default=8, help='number of hr channels to use')
    parser.add_argument('--ncha_clm', type=int, default=7, help='number of lr channels to use')
    parser.add_argument('-c', '--crop-size', help='image crop size', default=100, type=int)
    parser.add_argument('--ncha', type=int, default=7, help='number of lr channels to use')
    parser.add_argument('--ncha1', type=int, default=4, help='number of lr channels to use')
    parser.add_argument('--ncha2', type=int, default=3, help='number of lr channels to use')
    return parser.parse_args()


if __name__ == '__main__':
    """Tests Noise2Noise."""

    # Parse test parameters
    params = parse_args()

    # Initialize model and test
    n2n = Noise2Noise(params,trainable=False)
    params.redux = False
    params.clean_targets = True
    test_loader = load_dataset(params.data, 0, params, shuffled=False, single=True,drop_last=True)
    n2n.load_model(params.load_ckpt)
    n2n.test(test_loader, show=params.show_output)
