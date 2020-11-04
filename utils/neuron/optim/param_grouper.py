from neuron.config import registry


__all__ = ['ParamGrouper']


@registry.register_module
class ParamGrouper(object):

    def __init__(self, lr, weight_decay, special_lrs={},
                 special_wds={}):
        self.base_lr = lr
        self.base_wd = weight_decay
        self.special_lrs = special_lrs
        self.special_wds = special_wds
    
    def __call__(self, *modules):
        # collect all parameters and their names
        named_params = []
        for m in modules:
            named_params += list(m.named_parameters())
        
        # build param_groups
        param_groups = []
        for name, params in named_params:
            if not params.requires_grad:
                continue
            
            # learning rate
            lr_keys = [key for key in self.special_lrs if name in key]
            if len(lr_keys) == 0:
                lr = self.base_lr
            elif len(lr_keys) == 1:
                lr = self.special_lrs[lr_keys[0]]
            else:
                raise ValueError(
                    'The lr group of parameter {} is ambiguous since it'
                    'has multiple matches in special_lrs'.format(name))
            
            # weight decay
            wd_keys = [key for key in self.special_wds if name in key]
            if len(wd_keys) == 0:
                wd = self.base_wd
            elif len(wd_keys) == 1:
                wd = self.special_wds[wd_keys[0]]
            else:
                raise ValueError(
                    'The wd group of parameter {} is ambiguous since it'
                    'has multiple matches in special_wds'.format(name))
            
            # append to param_groups
            param_groups += [{'params': [params],
                              'lr': lr,
                              'weight_decay': wd}]
        
        return param_groups
