# # -- coding: utf-8 --**
from cProfile import label
from operator import mod
import os
# import sys
# sys.path.append("..")

import numpy as np
import torch
from torch.cuda.amp import autocast, GradScaler
import Models.model as model
import copy
import argparse
import json
import Utils.utils as utils
from Utils.loss_impl import *
import torch.nn as nn


class Train_Session(utils.Session):
    def _build_model(self):
        super()._build_model()
        model_config = self.config['Model']
        train_config = self.config['Train']

        # load model
        param_list = []
        self.net = nn.Sequential(
            getattr(model, model_config['arch'])()
        )
        self.net = self.net.to(self.device)
        param_list = [
            {'params':self.net.parameters(), 'lr':train_config['lr'], 'weight_decay':train_config['weight_decay']},]



        self.criterion = Loss()

        self.optimizer = torch.optim.Adam(param_list)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.2, patience=5)
      
        params = np.sum([p.numel() for p in self.net.parameters()]).item()
        self.logger.info(f"Loaded {model_config['name']} parameters : {params:.3e}")

    def _batch(self, item):
        with autocast():
            data, gt = item
            output = self.net(data.to(self.device))
            gt = gt.to(self.device)
            loss = self.criterion(output, gt)
            pred = output.argmax(1)

        if self.net.training:
            # loss.backward()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            # self.optimizer.step()
            self.optimizer.zero_grad()

        return {'loss':loss.cpu(),
                'pred': pred}

    # train/eval FSN
    def _epoch(self, data_loader, epoch=None):
        acc = utils.Param_Detector()
        loss = utils.Param_Detector()
        time = utils.Time_Detector()
        class_pred = utils.Category_Detector(9)
        
        for i, item in enumerate(data_loader):
            result = self._batch(item)
            loss.update(result['loss'])
            time.update(item[1].size(0))
            acc.update(result['pred'].cpu().eq(item[1].argmax(1)).sum(), item[1].size(0))
            class_pred.update(result['pred'].cpu(), item[1].argmax(1))

        cur_lr = self.optimizer.param_groups[-1]['lr']
        mode = 'train' if self.net.training else 'eval'
        
        if mode == 'train':
            self.logger.info(f'{mode} Epoch:{epoch}, loss:{loss.avg:.3f}, lr:{cur_lr:.2e}, {time.avg:.6f}  seconds/batch')
        else:
            self.logger.info(f'{mode} Acc:{acc.avg:.2f}, Epoch:{epoch}, loss:{loss.avg:.3f}, {time.avg:.6f}  seconds/batch')
        return {'loss': loss.avg.detach().numpy(),
                'time':time.avg}

    def train(self):
        # load the model
        self._build_model()

        # load the dataset
        train_loader = self._load_data('Train')
        eval_loader = self._load_data('Test')

        min_loss = float('inf')
        self.scaler = GradScaler()

        # trainning
        for epoch in range(1, self.config['Train']['num_epochs']+1):
            self.net.train()
            train_result = self._epoch(train_loader, epoch)

            self.net.eval()
            with torch.no_grad():
                eval_result = self._epoch(eval_loader, epoch)

            # record the result
            if self.writer:
                self.writer.add_scalar('Loss/train', train_result['loss'], epoch)
                self.writer.add_scalar('Loss/eval', eval_result['loss'], epoch)

            if epoch > 5:
                self.scheduler.step(eval_result['loss'])
            
            if eval_result['loss'] < min_loss:
                checkpoint = copy.deepcopy(self.net.state_dict())
                min_loss = eval_result['loss']
                best_epoch = epoch
                if self.config['Recorder']['save_log']:
                    checkpoint_path = os.path.join(self.log_dir, 'checkpoint.pkl')
            torch.save(checkpoint, checkpoint_path)

        self.logger.info(f'Min loss {min_loss:.3f} @ {best_epoch} epoch')

    def test(self):
        self.net.load_state_dict(torch.load(os.path.join(self.log_dir, 'checkpoint.pkl')))
        self.net.eval()
        with torch.no_grad():
            # test each scene

            test_loader = self._load_data('Test')
            test_result = self._epoch(test_loader)

            if self.writer:
                test_info = f"loss:{test_result['loss']:.3f}, {test_result['time']:.6f}  seconds/batch"
                self.writer.add_text('Test', test_info)

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--config', type=str, default='default')
    args.add_argument('--override', default=None, help='Arguments for overriding config')
    args = vars(args.parse_args())
    config = json.load(open(f"Config/{args['config']}.json", 'r'))

    if args['override'] is not None:
        override = args['override'].split(',')
        for item in override:
            key, value = item.split('=')
            if '.' in key:
                config[key.split('.')[0].strip()][key.split('.')[1].strip()] = value.strip()

    sess = Train_Session(config)
    sess.train()
    sess.test()
    sess.close()