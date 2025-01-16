# A Multi-task Learning Framework for Dual-polarization SAR Imagery Despeckling in Temporal Change Detection Scenarios

**Jie Li**, **Shaowei Shi**, **Liupeng Lin**, **Qiangqiang Yuan**, **Huanfeng Shen**, **Liangpei Zhang**

**Abstract**:
_The despeckling task for synthetic aperture radar (SAR) has long faced the challenge of obtaining clean images. Although unsupervised deep learning despeckling methods alleviate this issue, they often struggle to balance despeckling effectiveness and the preservation of spatial details. Furthermore, some unsupervised despeckling approaches overlook the effect of land cover changes when dual-temporal SAR images are used as training data. To address this issue, we propose a multitask learning framework for dual-polarization SAR imagery despeckling and change detection (MTDN). This framework integrates polarization decomposition mechanisms with dual-polarization SAR images, and utilizes a change detection network to guide and constrain the despeckling network for optimized performance. Specifically, the despeckling branch of this framework incorporates polarization and spatiotemporal information from dual-temporal dual-polarization SAR images to construct a despeckling network. It employs various attention mechanisms to recalibrate features across local/global, channel, and spatial dimensions, and before and after despeckling. The change detection branch, which combines Transformer and convolutional neural networks, helps the despeckling branch effectively filter out spatiotemporal information with substantial changes. The multitask joint loss function is weighted by the generated change detection mask to achieve collaborative optimization. Despeckling and change detection experiments are conducted using a dual-polarization SAR dataset to assess the effectiveness of the proposed framework. The despeckling experiments indicate that MTDN efficiently eliminates speckle noise while preserving polarization information and spatial details, and surpasses current leading SAR despeckling methods. The equivalent number of looks (ENL) for MTDN in the agricultural change area increased to 155.0630, and the edge detail preservation (EPD) metric improved to 0.9963. In contrast, the best-performing deSpeckNet among the comparison methods achieved ENL = 81.9933 and EPD = 0.9739 in the same region. Furthermore, the change detection experiments confirm that MTDN yields precise predictions, highlighting its exceptional capability in practical applications._

**Official Pytorch implementation for the paper accepted by ISPRS.**

<div style="display: flex; justify-content: space-between;">
    <img src="img/Comparison_results_1.png" alt="Denoising comparison 1" style="width: 48%;"/>
    <img src="img/Comparison_results_2.png" alt="Denoising comparison 2" style="width: 41%;"/>
</div>

## Resources

- [Arxiv]()
- [Conference]()
- [Supplementary]()

## Python Requirements

This code was tested on:

- Python 3.9
- Pytorch 1.0

## Sentinel-1_DATA
LINK: https://pan.baidu.com/s/1Z8fmNKd8j-sULqXrlzs1Ow?pwd=0116

## Preparing Training Dataset

Images in the training set are from the Sentinel-1 dataset with size 17700*8500 pixels.There are 22 time points in total. Use polsarPro to extract its covariance matrix and cut it into a training set of size 128*128.
The size of each data in the training set is [N=3, H=128, W=128, C=14], where N=1 is the training data: the first and second of three consecutive noise images (directly spliced ​​along the channel dimension), N=2 is the target data: the third of three consecutive noise images, and N=3 is the change detection label (T1 and T2) As shown below:
Train Dataset
N=1 T1 (4+3 channel) + T2 (4+3 channel) 
N=2 T3 (4+3 channel)
N=3 Change detection label (1 channel)

```bash
python train.py 
--train-dir=./../MTDN/data/train 
--valid-dir=./../MTDN/data
--ckpt-save-path=./../MTDN/ckpts
```
- optional arguments:
  - `train-dir` Path to the Train set
  - `valid-dir` Path to the Valid set
  - `ckpt-save-path` Path to save the training set
 
 - ## Training

To train a network, run:

```bash
python train.py 
--train-dir=./../MTDN/data/train 
--valid-dir=./../MTDN/data
--ckpt-save-path=./../MTDN/ckpts
--pre-cd-model=./../MTDN/models/changedetection/n2n-epoch100.pth
```
- selected optional arguments:
  - `train-dir` Path to the Train set
  - `valid-dir` Path to the Valid set
  - `ckpt-save-path` Path to save the training set
  - pre-cd-model=Pre-trained change detection model parameters
 
## Citations


