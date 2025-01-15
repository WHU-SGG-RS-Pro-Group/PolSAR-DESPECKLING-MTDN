import os, time
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
from torch.optim import Adam, lr_scheduler,SGD
from TSCD.TSCD import BASE_Transformer, get_scheduler
from TSCD.CDloss import cross_entropy
from DSDN import DSDN
from tool.utils import *
from TSCD.cd_optim import Ranger
from scipy.io import savemat
from tool.metric_tool import ConfuseMatrixMeter
from datetime import datetime
import torch
import torch.nn as nn
import sys
class Logger(object):
    def __init__(self, outfile):
        self.terminal = sys.stdout
        self.log_path = outfile
        now = time.strftime("%c")
        self.write('================ (%s) ================\n' % now)

    def write(self, message):
        self.terminal.write(message)
        with open(self.log_path, mode='a') as f:
            f.write(message)

    def write_dict(self, dict):
        message = ''
        for k, v in dict.items():
            message += '%s: %.7f ' % (k, v)
        self.write(message)

    def write_dict_str(self, dict):
        message = ''
        for k, v in dict.items():
            message += '%s: %s ' % (k, v)
        self.write(message)

    def flush(self):
        self.terminal.flush()
class Noise2Noise(object):
    """Implementation of Noise2Noise from Lehtinen et al. (2018)."""

    def __init__(self, params,  trainable):
        """Initializes model."""
        self.p = params
        self.trainable = trainable
        self.start_epoch = -1
        self._compile()
        self.running_metric = ConfuseMatrixMeter(n_class=2)

    def _compile(self):
        """Compiles model (architecture, loss function, optimizers, etc.)."""

        self.model = DSDN(args=self.p)
        self.cd_model=BASE_Transformer(input_nc=7, output_nc=2, token_len=4, resnet_stages_num=4,
                                         with_pos='learned', enc_depth=1, dec_depth=8)

        # CUDA support
        self.use_cuda = torch.cuda.is_available() and self.p.cuda
        print("usecuda:", self.use_cuda)

        # Set optimizer and loss, if in training mode
        if self.trainable:
            if bool(self.p.trained_model):
                if len(self.p.gpu)>1:
                    print("load checkpoint from:", self.p.trained_model)
                    checkpoint = torch.load(self.p.trained_model)
                    self.start_epoch = checkpoint['epoch']
                    self.model.load_state_dict(checkpoint['net1'])
                    self.cd_model.load_state_dict(checkpoint['cd_net'])
                    # CUDA support
                    self.model = self.model.cuda()

                    self.cd_model=self.cd_model.cuda()

                    self.model = torch.nn.DataParallel(self.model)

                    self.cd_model= torch.nn.DataParallel(self.cd_model)

                    self.optim = Adam(self.model.parameters(), lr=self.p.learning_rate)
                    self.cd_optim=SGD(self.cd_model.parameters(), lr=0.01,
                                        momentum=0.9,
                                        weight_decay=5e-4)

                    self.optim.load_state_dict(checkpoint['optimizer1'])
                    self.cd_optim.load_state_dict(checkpoint['cd_optim'])
                    self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optim,
                                                                    patience=self.p.nb_epochs / 4, factor=0.5,
                                                                    verbose=True)
                    self.CD_scheduler = get_scheduler(self.cd_optim)

                else:

                    print("load checkpoint from:", self.p.trained_model)
                    checkpoint = torch.load(self.p.trained_model)
                    self.start_epoch = checkpoint['epoch']
                    self.model.load_state_dict(checkpoint['net1'])

                    self.cd_model.load_state_dict(checkpoint['cd_net'])

                    # CUDA support
                    self.model = self.model.cuda()
                    self.cd_model = self.cd_model.cuda()
                    self.optim = Adam(self.model.parameters(), lr=self.p.learning_rate)
                    self.cd_optim = Ranger(
                        [
                            {'params': [param for name, param in self.cd_model.named_parameters()
                                        if name[-4:] == 'bias'],
                             'lr': 2 * self.p.learning_rate},
                            {'params': [param for name, param in self.cd_model.named_parameters()
                                        if name[-6:] == 'weight'],
                             'lr': self.p.learning_rate, 'weight_decay': 1e-4},
                            {'params': [param for name, param in self.cd_model.named_parameters()
                                        if name[-6:] != 'weight' and name[-4:] != 'bias'],
                             'lr': self.p.learning_rate
                             }], lr=self.p.learning_rate)
                    self.optim.load_state_dict(checkpoint['optimizer1'])
                    self.cd_optim.load_state_dict(checkpoint['cd_optim'])
                    self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optim,
                                                                    patience=self.p.nb_epochs / 4, factor=0.5,
                                                                    verbose=True)
            else:
                print("load checkpoint from:", self.p.pre_cd_model)
                checkpoint_pre = torch.load(self.p.pre_cd_model)
                self.cd_model.load_state_dict(checkpoint_pre['cd_net'])
                self.cd_model = self.cd_model.cuda()

                self.cd_model = torch.nn.DataParallel(self.cd_model)
                self.cd_optim = SGD(self.cd_model.parameters(), lr=0.01,
                                    momentum=0.9,
                                    weight_decay=5e-4)
                self.CD_scheduler = get_scheduler(self.cd_optim)
                # CUDA support
                self.model = self.model.cuda()

                if len(self.p.gpu )>1:
                    self.model = torch.nn.DataParallel(self.model)

                self.optim = Adam(self.model.parameters(),
                                  lr=self.p.learning_rate,
                                  betas=self.p.adam[:2],
                                  eps=self.p.adam[2])
                
                self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optim,
                                                                patience=self.p.nb_epochs / 4, factor=0.5, verbose=True)

            # Loss function
            if self.p.loss == 'hdr':
                assert self.is_mc, 'Using HDR loss on non Monte Carlo images'
                self.loss = HDRLoss()
            elif self.p.loss == 'l2':
                self.loss = nn.MSELoss()
            else:
                self.loss = nn.L1Loss()
            self.cd_loss = cross_entropy
        # CUDA support
        if self.use_cuda:
            if self.trainable:
                self.loss = self.loss.cuda()

    def _print_params(self):
        """Formats parameters to print when training."""

        print('Training parameters: ')
        self.p.cuda = self.use_cuda
        param_dict = vars(self.p)
        pretty = lambda x: x.replace('_', ' ').capitalize()
        print('\n'.join('  {} = {}'.format(pretty(k), str(v)) for k, v in param_dict.items()))

    def load_model(self, ckpt_fname):
        """Loads model from checkpoint file."""

        print('Loading checkpoint from: {}'.format(ckpt_fname))
        if self.use_cuda:
            checkpoint = torch.load(ckpt_fname)
            self.model.load_state_dict(checkpoint['net1'])
            self.model=self.model.cuda(0)
            self.cd_model.load_state_dict(checkpoint['cd_net'])
            self.cd_model=self.cd_model.cuda(0)

        else:
            self.model.module.load_state_dict(torch.load(ckpt_fname, map_location='cpu')['net1'])

    def _on_epoch_end(self, stats, train_loss,train_loss2, epoch, epoch_start, valid_loader):
        """Tracks and saves starts after each epoch."""
        train_log_filename = "train_log.txt"
        train_log_filepath = os.path.join(self.p.ckpt_save_path, train_log_filename)
        # Evaluate model on validation set
        print('\rTesting model on validation set... ', end='')
        epoch_time = time_elapsed_since(epoch_start)[0]
        valid_loss, valid_time, valid_psnr = self.eval(valid_loader)
        show_on_epoch_end(epoch_time, valid_time, valid_loss, valid_psnr)
        # Decrease learning rate if plateau
        self.scheduler.step(valid_loss)
        self.CD_scheduler.step()
        # Save checkpoint
        stats['train_loss'].append(train_loss)
        stats['valid_loss'].append(valid_loss)
        stats['valid_psnr'].append(valid_psnr)

        if 1:
            if not os.path.isdir(self.p.ckpt_save_path):
                os.mkdir(self.p.ckpt_save_path)
            ckpt_dir_name = f'{self.p.ckpt_name}'
            self.ckpt_dir = os.path.join(self.p.ckpt_save_path, ckpt_dir_name)
            if not os.path.isdir(self.ckpt_dir):
                os.mkdir(self.ckpt_dir)
            if self.p.dataparallel:
                checkpoint = {
                    'epoch': epoch,
                    'net1': self.model.module.state_dict(),
                    'cd_net':self.cd_model.module.state_dict(),
                    'optimizer1': self.optim.state_dict(),
                    'cd_optim':self.cd_optim.state_dict(),
                    'sche':self.scheduler.state_dict(),
                    'cd_sche': self.CD_scheduler.state_dict()
                }
                fname_resnet = '{}/n2n-epoch{}.pth'.format(self.ckpt_dir, epoch + 1)
                torch.save(checkpoint, fname_resnet)
                print('Saving datapallel checkpoint to: {}\n'.format(fname_resnet))
            else:
                checkpoint = {
                    'epoch': epoch,
                    'net1': self.model.state_dict(),
                    'cd_net':self.cd_model.state_dict(),
                    'optimizer1': self.optim.state_dict(),
                    'cd_optim':self.cd_optim.state_dict(),
                    'sche':self.scheduler.state_dict(),
                    'cd_sche': self.CD_scheduler.state_dict()
                }
                fname_resnet = '{}/n2n-epoch{}.pth'.format(self.ckpt_dir, epoch + 1)
                torch.save(checkpoint, fname_resnet)
                print('Saving checkpoint to: {}\n'.format(fname_resnet))
        train_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [T_Loss] {loss_str1}[T_Loss2] {loss_str3}[V_Loss] {loss_str2}[psnr] {psnr}\n"
        to_write = train_log_txt_formatter.format(time_str=time.strftime("%Y_%m_%d_%H:%M:%S"),
                                                  epoch=epoch+1,
                                                  loss_str1=" ".join(["{}".format(train_loss)]),
                                                  loss_str3=" ".join(["{}".format(train_loss2)]),

                                                  loss_str2=" ".join(["{}".format(valid_loss)]),
                                                  psnr=" ".join(["{}".format(valid_psnr)]))
        with open(train_log_filepath, "a") as f:
            f.write(to_write)

    def test(self, test_loader, show):
        """Evaluates denoiser on test set."""

        self.model.train(False)
        label_cd=[]
        pre_cd=[]
        source_imgs = []
        denoised_imgs = []
        clean_imgs = []
        # Create directory for denoised images
        denoised_dir = os.path.dirname(self.p.data)
        save_path = os.path.join(denoised_dir, 'denoised')
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        elapsed_time = np.zeros((1000, 1), dtype=np.float64)

        for batch_idx, source in enumerate(test_loader):
            if show == 0 or batch_idx >= show:
                break

            source_1 = source.narrow(1, 0, 7).cuda(0)
            source_2 = source.narrow(1, 7, 7).cuda(0)
            start_time = time.time()
            output, x_mid = self.cd_model(source_1, source_2)
            end_time = time.time()
            source_de=source.narrow(1,0,14).cuda(0)
            pred = torch.argmax(output, dim=1, keepdim=True)
            target = source
            source_imgs.append(source_de)
            clean_imgs.append(target)
            start_time = time.time()
            if self.use_cuda:
                source = source.cuda(0)

            # Denoise
            denoised_img = self.model(source_de,x_mid).detach()
            end_time = time.time()
            denoised_imgs.append(denoised_img)
            pre_cd.append(pred)
            G_pred = output.detach()

            G_pred = torch.argmax(G_pred, dim=1)


        # Squeeze tensors
        source_imgs = [t.squeeze(0) for t in source_imgs]
        denoised_imgs = [t.squeeze(0) for t in denoised_imgs]
        clean_imgs = [t.squeeze(0) for t in clean_imgs]
        pre_cd=[t.squeeze(0) for t in pre_cd]
        # Create montage and save images
        print('Saving images and montages to: {}'.format(save_path))
        for i in range(len(source_imgs)):
            img_name = test_loader.dataset.imgs[i]
            create_montage(img_name, save_path,  denoised_imgs[i],pre_cd[i])
        savemat(os.path.join(denoised_dir, 'rdn_time.mat'), {'data': elapsed_time})

        with open(save_path + r"\tckpt.txt", "w") as f:
            f.write(self.p.load_ckpt)
        print('all done!')

    def eval(self, valid_loader):
        """Evaluates denoiser on validation set."""

        self.model.train(False)
        self.cd_model.train(False)
        with torch.no_grad():
            valid_start = datetime.now()
            loss_meter = AvgMeter()
            psnr_meter = AvgMeter()
            loss_v = []
            loss_deno = []

            for batch_idx, (source, target, label) in enumerate(valid_loader):
                if self.use_cuda:
                    source = source.cuda()
                    target = target.cuda()
                    label = label.cuda()

                source_de = source.index_select(1, torch.tensor(new_order).cuda())
                target_de = target.narrow(1, 0, self.p.ncha)
                source_1 = source_de.narrow(1, 0, 7)
                source_2 = source_de.narrow(1, 7, 7)
                label = label.narrow(1, 0, 1)

                output, x_mid = self.cd_model(source_1, source_2)

                loss_cd = self.cd_loss(output, label)

                source_denoised = self.model(source_de, x_mid)
                pred = torch.argmax(output, dim=1, keepdim=True)
                with torch.no_grad():
                    inverse_label = 1 - pred
                loss_de = self.loss(source_denoised * inverse_label, target_de * inverse_label)
                loss_v.append(loss_cd.item())
                loss_deno.append(loss_de.item())
                loss = loss_cd * 10 + loss_de
                loss_meter.update(loss.item())
                target_l = label.detach()
                G_pred = output.detach()

                G_pred = torch.argmax(G_pred, dim=1)

                current_score = self.running_metric.update_cm(pr=G_pred.cpu().numpy(), gt=target_l.cpu().numpy())

                if loss.item() > 1:
                    print("valid loss too large,it is :", loss.item())

                loss_meter.update(loss.item())

                # TODO: Find a way to offload to GPU, and deal with uneven batch sizes
                for i in range(self.p.batch_size):
                    source_denoised = source_denoised.cpu()

                    target_de = target_de.cpu()

            print("eval_cd:", np.mean(loss_v))
            print("eval_de:", np.mean(loss_deno))
            scores = self.running_metric.get_scores()
            epoch_acc = scores['mf1']
            self.logger.write('epoch_mf1= %.5f\n' %
                              epoch_acc)
            message = ''
            for k, v in scores.items():
                message += '%s: %.5f ' % (k, v)
            self.logger.write(message + '\n')
            self.logger.write('\n')
            valid_loss = loss_meter.avg
            valid_time = time_elapsed_since(valid_start)[0]
            psnr_avg = psnr_meter.avg

        return valid_loss, valid_time, psnr_avg

    def train(self, train_loader, valid_loader):
        """Trains denoiser on training set."""
        logger_path = os.path.join(self.p.ckpt_save_path, 'log.txt')
        self.logger = Logger(logger_path)
        train_log_filename = "CD_params.txt"
        train_log_filepath = os.path.join(self.p.ckpt_save_path, train_log_filename)
        self.model.train(True)
        self.cd_model.train(True)
        self._print_params()
        with torch.no_grad():
            num_batches = len(train_loader)

            stats = {'train_loss': [],
                     'valid_loss': [],
                     'valid_psnr': []}

            # Main training loop
            train_start = datetime.now()
            # TODO:CAN BE DELETED

        for epoch in range(self.start_epoch + 1, self.p.nb_epochs):
            print('EPOCH {:d} / {:d}'.format(epoch + 1, self.p.nb_epochs))
            with torch.no_grad():
                epoch_start = datetime.now()
                train_loss_meter = AvgMeter()
                loss_meter = AvgMeter()
                time_meter = AvgMeter()
                loss_v = []
                loss_deno = []
                largelosscount = 0
            for batch_idx, (source, target, label) in enumerate(train_loader):
                if self.use_cuda:
                    source = source.cuda()
                    target = target.cuda()
                    label = label.cuda()

                batch_start = datetime.now()
                progress_bar(batch_idx, num_batches, self.p.report_interval, loss_meter.val)

                source_de = source.index_select(1, torch.tensor(new_order).cuda())
                target_de = target.narrow(1, 0, self.p.ncha)
                source_1 = source_de.narrow(1, 0, 7)
                source_2 = source_de.narrow(1, 7, 7)

                label = label.narrow(1, 0, 1)

                output, x_mid = self.cd_model(source_1, source_2)

                loss_cd = self.cd_loss(output, label)
                pred = torch.argmax(output, dim=1, keepdim=True)
                source_denoised = self.model(source_de, x_mid)
                with torch.no_grad():
                    inverse_label = 1 - pred
                loss_de = self.loss(source_denoised * inverse_label, target_de * inverse_label)
                loss = loss_cd * 10 + loss_de

                loss_v.append(loss_cd.item())
                loss_deno.append(loss_de.item())
                loss_meter.update(loss.item())

                self.optim.zero_grad()
                self.cd_optim.zero_grad()
                loss.backward()
                self.optim.step()
                self.cd_optim.step()

                with torch.no_grad():

                    time_meter.update(time_elapsed_since(batch_start)[1])
                    if (batch_idx + 1) % self.p.report_interval == 0 and batch_idx:
                        show_on_report(batch_idx, num_batches, loss_meter.avg, time_meter.avg)
                        train_loss_meter.update(loss_meter.avg)
                        loss_meter.reset()
                        time_meter.reset()
                    target = label.detach()
                    G_pred = output.detach()

                    G_pred = torch.argmax(G_pred, dim=1)

                    current_score = self.running_metric.update_cm(pr=G_pred.cpu().numpy(), gt=target.cpu().numpy())
            print("train_cd:", np.mean(loss_v))
            print("train_de:", np.mean(loss_deno))
            if 1:
                scores = self.running_metric.get_scores()
                epoch_acc = scores['mf1']
                self.logger.write('Epoch %d ,epoch_mf1= %.5f\n' %
                                  (epoch + 1, epoch_acc))
                message = ''
                for k, v in scores.items():
                    message += '%s: %.5f ' % (k, v)
                self.logger.write(message + '\n')
                self.logger.write('\n')
                train_log_txt_formatter = " [Epoch] {epoch:03d} [acc] {acc}[loss1] {loss1}[loss3] {loss3}\n"
                to_write = train_log_txt_formatter.format(epoch=epoch + 1,

                                                          acc=" ".join(["{}".format(np.mean(current_score))]),
                                                          loss1=" ".join(["{}".format(np.mean(loss_v))]),
                                                          # loss2=" ".join(["{}".format(np.mean(loss_di))]),
                                                          loss3=" ".join(["{}".format(np.mean(loss_deno))]))
                with open(train_log_filepath, "a") as f:
                    f.write(to_write)
            with torch.no_grad():
                # Epoch end, save and reset tracker
                self._on_epoch_end(stats, train_loss_meter.avg, np.mean(loss_deno), epoch, epoch_start, valid_loader)
                train_loss_meter.reset()
                self.optim.zero_grad()
                self.cd_optim.zero_grad()
                torch.cuda.empty_cache()

        train_elapsed = time_elapsed_since(train_start)[0]
        print('Training done! Total elapsed time: {}\n'.format(train_elapsed))

