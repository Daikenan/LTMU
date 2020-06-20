import torch
import torch.nn as nn

from neuron.config import registry


__all__ = ['MultiTaskLoss', 'ReID_Loss']


@registry.register_module
class MultiTaskLoss(nn.Module):

    def __init__(self, **kwargs):
        assert len(kwargs) > 0
        super(MultiTaskLoss, self).__init__()
        self.losses = {}
        for k, v in kwargs.items():
            self.losses[k] = (v.loss, v.weight)
    
    def forward(self, input, target):
        loss = {}
        for name, (loss_fn, weight) in self.losses.items():
            loss_i = loss_fn(input, target)
            if isinstance(loss_i, torch.Tensor):
                loss[name] = loss_i
            elif isinstance(loss_i, dict):
                loss[name] = loss_i.pop('loss')
                for k, v in loss_i.items():
                    if k in loss:
                        raise KeyError(
                            'The criterion name {} is ambiguous since'
                            'it appears in multiple losses'.format(k))
                    else:
                        loss[k] = v
            else:
                raise ValueError(
                    'Expected the output of loss {} to be a Tensor'
                    'or a dict, but got {}'.format(
                        name, loss_i.__class__.__name__))
            loss['loss'] += loss[name] * weight
        return loss


@registry.register_module
class ReID_Loss(MultiTaskLoss):

    def __init__(self, **kwargs):
        for name in kwargs:
            assert 'cls' in name or 'rank' in name
        super(ReID_Loss, self).__init__(**kwargs)

    def forward(self, *args):
        if len(args) == 2:
            feats, labels = args
        elif len(args) == 3:
            scores, feats, labels = args
        else:
            raise ValueError(
                'Expected 2 or 3 inputs, but got {}'.format(len(args)))

        loss = {'loss': 0.}
        for name, (loss_fn, weight) in self.losses.items():
            if 'cls' in name:
                if len(args) == 2:
                    continue
                loss_i = loss_fn(scores, labels)
            elif 'rank' in name:
                loss_i = loss_fn(feats, labels)
            else:
                raise KeyError('Unsupport loss {}'.format(name))
            
            if isinstance(loss_i, torch.Tensor):
                loss[name] = loss_i
            elif isinstance(loss_i, dict):
                loss[name] = loss_i.pop('loss')
                for k, v in loss_i.items():
                    if k in loss:
                        raise KeyError(
                            'The criterion name {} is ambiguous since'
                            'it appears in multiple losses'.format(k))
                    else:
                        loss[k] = v
            else:
                raise ValueError(
                    'Expected the output of loss {} to be a Tensor'
                    'or a dict, but got {}'.format(
                        name, loss_i.__class__.__name__))

            loss['loss'] += loss[name] * weight
        return loss
