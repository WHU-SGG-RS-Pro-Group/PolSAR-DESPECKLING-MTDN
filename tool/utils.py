import numpy as np
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import scipy.io as sio
import os
from datetime import datetime
import OpenEXR
import torch
import Imath
def clear_line():
    """Clears line from any characters."""

    print('\r{}'.format(' ' * 80), end='\r')

def create_montage(img_name, save_path,  denoised_t, pre_cd):
    """Creates montage for easy comparison."""

    fig, ax = plt.subplots(1, 3, figsize=(9, 3))
    fig.canvas.manager.set_window_title(img_name.capitalize()[:-4])

    denoised_t = denoised_t.cpu()
    pre_cd = pre_cd.cpu()

    # Save to files

    fname = os.path.splitext(img_name)[0]
    sio.savemat(os.path.join(save_path, f'{fname}-proposed_denoised.mat'), {'data': np.array(denoised_t)})
    sio.savemat(os.path.join(save_path, f'pred{fname}-proposed_denoised.mat'), {'data': np.array(pre_cd)})
    print('test saved!')

def progress_bar(batch_idx, num_batches, report_interval, train_loss):
    """Neat progress bar to track training."""

    dec = int(np.ceil(np.log10(num_batches)))
    bar_size = 21 + dec
    progress = (batch_idx % report_interval) / report_interval
    fill = int(progress * bar_size) + 1
    print('\rBatch {:>{dec}d} [{}{}] Train loss: {:>1.5f}'.format(batch_idx + 1, '=' * fill + '>',
                                                                  ' ' * (bar_size - fill), train_loss, dec=str(dec)),
          end='')
def time_elapsed_since(start):
    """Computes elapsed time since start."""

    timedelta = datetime.now() - start
    string = str(timedelta)[:-7]
    ms = int(timedelta.total_seconds() * 1000)

    return string, ms
def load_hdr_as_tensor(img_path):
    """Converts OpenEXR image to torch float tensor."""

    # Read OpenEXR file
    if not OpenEXR.isOpenExrFile(img_path):
        raise ValueError(f'Image {img_path} is not a valid OpenEXR file')
    src = OpenEXR.InputFile(img_path)
    pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
    dw = src.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    # Read into tensor
    tensor = torch.zeros((3, size[1], size[0]))
    for i, c in enumerate('RGB'):
        rgb32f = np.fromstring(src.channel(c, pixel_type), dtype=np.float32)
        tensor[i, :, :] = torch.from_numpy(rgb32f.reshape(size[1], size[0]))

    return tensor

def show_on_epoch_end(epoch_time, valid_time, valid_loss, valid_psnr):
    """Formats validation error stats."""

    clear_line()
    print('Train time: {} | Valid time: {} | Valid loss: {:>1.5f} | Avg PSNR: {:.2f} dB'.format(epoch_time, valid_time,
                                                                                                valid_loss, valid_psnr))
def show_on_report(batch_idx, num_batches, loss, elapsed):
    """Formats training stats."""

    clear_line()
    dec = int(np.ceil(np.log10(num_batches)))
    print('Batch {:>{dec}d} / {:d} | Avg loss: {:>1.5f} | Avg train time / batch: {:d} ms'.format(batch_idx + 1,
                                                                                                  num_batches, loss,
                                                                                                  int(elapsed),
                                                                                                  dec=dec))

class AvgMeter(object):
    """Computes and stores the average and current value.
    Useful for tracking averages such as elapsed times, minibatch losses, etc.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0.
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def de_norm(tensor_data):
    return tensor_data * 0.5 + 0.5