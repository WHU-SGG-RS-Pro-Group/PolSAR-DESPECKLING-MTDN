#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from argparse import ArgumentParser
from tool.datasets import load_dataset
from noise2noise import Noise2Noise
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import warnings


warnings.filterwarnings("ignore")

def parse_args():
    """Command-line argument parser for training."""

    # New parser
    parser = ArgumentParser(description='PyTorch implementation of Noise2Noise from Lehtinen et al. (2018)')
    parser.add_argument('-d', '--dataparallel', help='', default=0, type=int)
    parser.add_argument('--gpu', default='0', type=str, help='gpu id')

    # Data parameters
    parser.add_argument('-t', '--train-dir', help='training set path', default=r'./../MTDN/data/train')
    parser.add_argument('-v', '--valid-dir', help='test set path', default=r'./../MTDN/data')
    parser.add_argument('--ckpt-save-path', help='checkpoint save path', default='./../MTDN/ckpts')
    parser.add_argument('--report-interval', help='batch report interval', default=5000, type=int)
    parser.add_argument('--trained-model', help='keep on training baesd on trained model', default=r'', type=str)
    parser.add_argument('--ckpt-name', help='saved checkpoint name prefix', default='denoise-multi2-7cha-new', type=str)
    parser.add_argument('--pre-cd-model', help='', default=r'./../MTDN/models/changedetection/n2n-epoch100.pth', type=str)
    # Training hyperparameters
    parser.add_argument('-lr', '--learning-rate', help='learning rate', default=0.001, type=float)
    parser.add_argument('-a', '--adam', help='adam parameters', nargs='+', default=[0.9, 0.99, 1e-8], type=list)
    parser.add_argument('-b', '--batch-size', help='minibatch size', default=20, type=int)
    parser.add_argument('-e', '--nb-epochs', help='number of epochs', default=100, type=int)
    parser.add_argument('-l', '--loss', help='loss function', choices=['l1', 'l2', 'hdr'], default='l2', type=str)
    parser.add_argument('--cuda', help='use cuda', action='store_true',default=True)
    parser.add_argument('-c', '--crop-size', help='random crop size', default=100, type=int)
    parser.add_argument('--nDense', type=int, default=6, help='nDenselayer of RDB')
    parser.add_argument('--growthRate', type=int, default=32, help='growthRate of dense net')
    parser.add_argument('--nFeat', type=int, default=64, help='number of feature maps')
    parser.add_argument('--ncha_modis', type=int, default=4, help='number of hr channels to use')
    parser.add_argument('--ncha_clm', type=int, default=8, help='number of lr channels to use')
    parser.add_argument('--ncha', type=int, default=7, help='number of lr channels to use')
    return parser.parse_args()

if __name__ == '__main__':
    """Trains Noise2Noise."""

    # Parse training parameters

    params = parse_args()
    # Train/valid datasets
    train_loader = load_dataset(params.train_dir,  params=params, shuffled=True,drop_last=True)
    valid_loader = load_dataset(params.valid_dir,  params=params, shuffled=False,drop_last=True)


    # Initialize model and train
    n2n = Noise2Noise(params, trainable=True)
    n2n.train(train_loader, valid_loader)
