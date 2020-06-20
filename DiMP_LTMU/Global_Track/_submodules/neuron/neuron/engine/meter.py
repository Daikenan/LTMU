import addict


__all__ = ['Meter']


class _AverageMeter(object):

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val):
        self.val = val
        self.sum += val
        self.count += 1
        self.avg = self.sum / self.count
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def __repr__(self):
        return '{:.3f}'.format(self.avg)


class Meter(addict.Dict):
    
    def update(self, metrics):
        for k, v in metrics.items():
            if k in self.keys():
                self.__getitem__(k).update(v)
            else:
                meter = _AverageMeter()
                meter.update(v)
                self.__setitem__(k, meter)
    
    def reset(self):
        for k in self.keys():
            self.__getitem__(k).reset()
