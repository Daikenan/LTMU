from torch.optim.lr_scheduler import _LRScheduler
from bisect import bisect_right

from neuron.config import registry


__all__ = ['WarmupMultiStepLR']


@registry.register_module
class WarmupMultiStepLR(_LRScheduler):

    def __init__(self,
                 optimizer,
                 steps,
                 gamma,
                 warmup_factor,
                 warmup_iters,
                 warmup_method,
                 last_epoch=-1):
        if steps != sorted(steps):
            raise ValueError(
                'Steps should be a list of increasing integers,'
                ' but got {}'.format(steps))
        assert warmup_method in ['constant', 'linear']
        self.steps = steps
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == 'constant':
                warmup_factor = self.warmup_factor
            elif self.warmup_method == 'linear':
                alpha = self.last_epoch / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [base_lr * warmup_factor * self.gamma ** \
            bisect_right(self.steps, self.last_epoch)
            for base_lr in self.base_lrs]
