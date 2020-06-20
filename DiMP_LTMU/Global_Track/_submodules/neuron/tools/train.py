from neuron.config import Config
from neuron.engine import Trainer


if __name__ == '__main__':
    config_file = 'configs/reid_baseline.yaml'
    cfg = Config.load(config_file)

    trainer = Trainer(cfg)
    trainer.train()
    trainer.test()
