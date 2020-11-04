import torch
import os
import os.path as osp
from datetime import datetime

import neuron.ops as ops
from neuron.config import registry
from .meter import Meter
from .logger import Logger


class Trainer(object):

    def __init__(self, cfg):
        # experimental information
        self.name = '{}_{}'.format(
            cfg.experiment,
            datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        self.work_dir = osp.join(cfg.work_dir, self.name)
        if not osp.exists(self.work_dir):
            os.makedirs(self.work_dir)
        self.log_level = cfg.log_level
        self.train_echo = cfg.train_echo
        self.val_echo = cfg.val_echo
        self.max_epochs = cfg.max_epochs
        self.start_epoch = cfg.start_epoch
        self.save_frequency = cfg.save_frequency
        self.val_frequency = cfg.val_frequency

        # GPU usage
        cuda = torch.cuda.is_available()
        cfg.use_gpu = cfg.use_gpu if cuda else False
        self.device = torch.device('cuda:0' if cfg.use_gpu else 'cpu')
        # store configurations
        self.cfg = cfg.deepcopy()

        # build model, loss and metrics
        self.model = registry.build(cfg.model).to(self.device)
        self.criterion = registry.build(cfg.loss).to(self.device)
        if cfg.metric:
            self.metric = registry.build(cfg.metric).to(self.device)
        else:
            self.metric = None

        # build optimizer
        if cfg.param_grouper:
            param_grouper = registry.build(cfg.param_grouper)
            params = param_grouper(self.model, self.criterion)
        else:
            params = list(self.model.parameters()) + \
                list(self.criterion.parameters())
            params = [p for p in params if p.requires_grad]
        cfg.optimizer.update({'params': params})
        self.optimizer = registry.build(cfg.optimizer)

        # build lr scheduler
        if cfg.lr_scheduler:
            cfg.lr_scheduler.update({'optimizer': self.optimizer})
            self.lr_scheduler = registry.build(cfg.lr_scheduler)
        else:
            self.lr_scheduler = None
        
        # build training, validation and test data
        self.train_data = self._build_dataset(cfg.train_data)
        self.val_data = self._build_dataset(cfg.val_data)
        self.test_data = self._build_dataset(cfg.test_data)

        # build evaluator
        if cfg.evaluator:
            cfg.evaluator.update({'dataset': self.test_data})
            self.evaluator = registry.build(cfg.evaluator)
        else:
            self.evaluator = None
        
        # state variables
        self._train_epoch = -1
        self._train_iter = -1
        self._val_epoch = -1
        self._val_iter = -1
        self._train_meters = Meter()
        self._val_meters = Meter()
        self._logger = Logger(
            self.work_dir, self.name, log_level=self.log_level)

    def train(self):
        self._hook_started()
        for epoch in range(self.start_epoch, self.max_epochs):
            self.train_epoch(self.train_data, epoch=epoch)
            if ((epoch + 1) % self.val_frequency == 0 or \
                (epoch + 1) == self.max_epochs) and \
                self.val_data is not None:
                self.val_epoch(self.val_data, epoch=epoch)
        self._hook_completed()
    
    @torch.enable_grad()
    def train_epoch(self, train_data, epoch):
        self._set_train(True)
        self._hook_epoch_started(epoch=epoch)
        for it, batch in enumerate(train_data):
            self._hook_iter_started(iter_=it)
            loss = self.train_step(batch)
            self._hook_iter_completed(metrics=loss)
        self._hook_epoch_completed()

    @torch.enable_grad()
    def train_step(self, batch):
        self._set_train(True)
        batch = ops.put_device(batch, self.device)

        # forward pass
        output = ops.adaptive_apply(self.model.forward_train, batch)
        
        # evaluate loss and ensure it to be a dictionary
        loss = ops.adaptive_apply(self.criterion, output)
        if isinstance(loss, torch.Tensor):
            loss = {'loss': loss}
        assert isinstance(loss, dict)

        # optimization step
        self.optimizer.zero_grad()
        loss['loss'].backward()
        self.optimizer.step()

        # evaluate metrics
        if self.metric is not None:
            metrics = ops.adaptive_apply(
                self.metric, ops.detach(output))
            loss.update(metrics)

        # convert to CPU arrays or scalars
        for k, v in loss.items():
            if not isinstance(v, torch.Tensor):
                continue
            if v.numel() == 1:
                loss[k] = v.item()
            else:
                loss[k] = v.detach().cpu().numpy()

        return loss
    
    @torch.no_grad()
    def val_epoch(self, val_data, epoch):
        self._set_train(False)
        self._hook_val_epoch_started(epoch=epoch)
        for it, batch in enumerate(val_data):
            self._hook_val_iter_started(iter_=it)
            loss = self.val_step(batch)
            self._hook_val_iter_completed(metrics=loss)
        self._hook_val_epoch_completed()
    
    @torch.no_grad()
    def val_step(self, batch):
        self._set_train(False)
        batch = ops.put_device(batch, self.device)

        # forward pass
        output = ops.adaptive_apply(self.model.forward_val, batch)
        
        # evaluate loss and ensure it to be a dictionary
        loss = ops.adaptive_apply(self.criterion, output)
        if isinstance(loss, torch.Tensor):
            loss = {'loss': loss}
        assert isinstance(loss, dict)

        # evaluate metrics
        if self.metric is not None:
            metrics = ops.adaptive_apply(
                self.metric, ops.detach(output))
            loss.update(metrics)

        # convert to CPU arrays or scalars
        for k, v in loss.items():
            if not isinstance(v, torch.Tensor):
                continue
            if v.numel() == 1:
                loss[k] = v.item()
            else:
                loss[k] = v.detach().cpu().numpy()

        return loss
    
    @torch.no_grad()
    def test(self):
        self._set_train(False)
        if self.evaluator is None:
            raise ValueError('The evaluator is not configured')
        self.evaluator.run(self.model)
        self.evaluator.report(self.model.name)
    
    def _set_train(self, flag=True):
        if flag:
            self.model.train()
            self.criterion.train()
        else:
            self.model.eval()
            self.criterion.eval()
    
    def _build_dataset(self, cfg):
        if not cfg:
            return None
        
        if 'DataLoader' in cfg.type:
            dataset = registry.build(cfg.dataset)
            cfg.update({'dataset': dataset})
            if 'sampler' in cfg:
                if 'shuffle' in cfg:
                    cfg.shuffle = False
                cfg.sampler.update({'dataset': dataset})

        return registry.build(cfg)
    
    def _adaptive_apply(self, func, args):
        if isinstance(args, (tuple, list)):
            return func(*args)
        elif isinstance(args, dict):
            return func(**args)
        else:
            return func(args)
    
    def _hook_started(self):
        self._logger.log(
            'Experiment[{}]: Training started...'.format(self.name))
        self._logger.log('Configurations:\n  {}'.format(repr(self.cfg)))
    
    def _hook_completed(self):
        self._logger.log(
            'Experiment[{}]: Training completed!'.format(self.name))

    def _hook_epoch_started(self, epoch):
        # store epoch
        self._train_epoch = epoch
        # update lr if applicable
        if self.lr_scheduler:
            self.lr_scheduler.step(epoch=epoch)
        # log text
        self._logger.log('Epoch[{}/{}]: Training started...'.format(
            epoch + 1, self.max_epochs))
    
    def _hook_epoch_completed(self):
        # log text
        self._logger.log('Epoch[{}/{}]: Training completed!'.format(
            self._train_epoch + 1, self.max_epochs))
        text = '\t'
        for name in self.train_echo:
            if not name in self._train_meters:
                continue
            text += ' {}: {:.3f}'.format(
                name, self._train_meters[name].avg)
        self._logger.log(text)
        # log metrics
        for k, v in self._train_meters.items():
            self._logger.add_scalar(
                'epoch_train/' + k, v.avg,
                global_step=self._train_epoch)
        # save model
        if (self._train_epoch + 1) % self.save_frequency == 0 or \
            (self._train_epoch + 1) == self.max_epochs:
            model_file = osp.join(
                self.work_dir,
                'model_{}.pth'.format(self._train_epoch + 1))
            torch.save(self.model, model_file)
        # reset meters
        self._train_meters.reset()
    
    def _hook_iter_started(self, iter_):
        # store iter
        self._train_iter = iter_
        # update lr if applicable
        if getattr(self.lr_scheduler, 'step_iter', None) is not None:
            self.lr_scheduler.step_iter(iter_=iter_)
    
    def _hook_iter_completed(self, metrics):
        # log text
        text = 'Epoch[{}/{}] Iter[{}/{}]'.format(
            self._train_epoch + 1, self.max_epochs,
            self._train_iter + 1, len(self.train_data))
        for name in self.train_echo:
            if not name in metrics:
                continue
            text += ' {}: {:.3f}'.format(name, metrics[name])
        self._logger.log(text)
        # log metrics
        for k, v in metrics.items():
            self._logger.add_scalar('iter_train/' + k, v)
        # update meters
        self._train_meters.update(metrics)
    
    def _hook_val_epoch_started(self, epoch):
        # store epoch
        self._val_epoch = epoch
        # log text
        self._logger.log(
            'Val Epoch[{}/{}]: Validation started...'.format(
                epoch + 1, self.max_epochs))
    
    def _hook_val_epoch_completed(self):
        # log text
        self._logger.log(
            'Val Epoch[{}/{}]: Validation completed!'.format(
                self._val_epoch + 1, self.max_epochs))
        text = '\t'
        for name in self.val_echo:
            if not name in self._val_meters:
                continue
            text += ' {}: {:.3f}'.format(
                name, self._val_meters[name].avg)
        self._logger.log(text)
        # log metrics
        for k, v in self._val_meters.items():
            self._logger.add_scalar(
                'epoch_val/' + k, v.avg,
                global_step=self._val_epoch)
        # reset meters
        self._val_meters.reset()
    
    def _hook_val_iter_started(self, iter_):
        # store iter
        self._val_iter = iter_
    
    def _hook_val_iter_completed(self, metrics):
        # log text
        text = 'Val Epoch[{}/{}] Iter[{}/{}]'.format(
            self._val_epoch + 1, self.max_epochs,
            self._val_iter + 1, len(self.val_data))
        for name in self.val_echo:
            if not name in metrics:
                continue
            text += ' {}: {:.3f}'.format(name, metrics[name])
        self._logger.log(text)
        # log metrics
        for k, v in metrics.items():
            self._logger.add_scalar('iter_val/' + k, v)
        # update meters
        self._val_meters.update(metrics)
    
    def _hook_exception_raised(self):
        pass
