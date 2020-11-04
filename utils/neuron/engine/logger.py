import os.path as osp
import logging
import sys
from tensorboardX import SummaryWriter


class Logger(object):
    LOG_LEVELS = {
        'CRITICAL': logging.CRITICAL,
        'ERROR': logging.ERROR,
        'WARNING': logging.WARNING,
        'INFO': logging.INFO,
        'DEBUG': logging.DEBUG,
        'NOTSET': logging.NOTSET}

    def __init__(self, log_dir, run_name, **kwargs):
        # console and file logger
        log_level = kwargs.pop('log_level', logging.DEBUG)
        if isinstance(log_level, str):
            log_level = self.LOG_LEVELS[log_level.upper()]
        fmt = logging.Formatter(
            '%(asctime)s %(levelname)s: %(message)s')
        self._logger = logging.getLogger(run_name)
        self._logger.setLevel(log_level)
        # console handler
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(log_level)
        ch.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
        self._logger.addHandler(ch)
        # file logger
        log_file = osp.join(log_dir, run_name + '.log')
        fh = logging.FileHandler(log_file)
        fh.setLevel(log_level)
        fh.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)s: %(message)s'))
        self._logger.addHandler(fh)

        # summary writer
        self._writer = SummaryWriter(logdir=log_dir)
    
    def log(self, msg, level=logging.INFO):
        self._logger.log(level, msg)
    
    def __getattr__(self, name):
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            if name in dir(self._writer):
                return object.__getattribute__(self._writer, name)
            elif name in dir(self._logger):
                object.__getattribute__(self._logger, name)
            else:
                raise AttributeError(
                    '\'{}\' has no attribute \'{}\''.format(
                        self.__class__.__name__, name))
    
    def close(self):
        self._logger.close()
        self._writer.close()
