import unittest
import os

import neuron.data as data
from neuron.config import Config, registry


class TestConfig(unittest.TestCase):

    def test_config(self):
        cfg = Config({'param1': 1}, param2=2, param3={'param4': 4})

        # check saving and loading
        for ext in ['.json', '.yaml', '.yml']:
            cfg_file = 'config' + ext
            cfg.dump(cfg_file)
            new_cfg = Config.load(cfg_file)
            os.remove(cfg_file)
            self.assertEqual(cfg, new_cfg)
        
        # check copying and merging
        new_cfg = cfg.deepcopy()
        self.assertNotEqual(id(cfg), id(new_cfg))
        new_cfg.update({'param1': 10})
        self.assertEqual(cfg.merge_from(new_cfg), new_cfg)
        self.assertEqual(cfg.merge_to(new_cfg), cfg)

        # check freezing and defrosting
        cfg.freeze()
        self.assertTrue(cfg.is_frozen)
        cfg.defrost()
        self.assertFalse(cfg.is_frozen)
    
    def test_registry(self):
        # check building modules
        cfg = Config(type='torch.Conv2d', in_channels=3,
                     out_channels=64, kernel_size=3, padding=1)
        module = registry.build(cfg)

        # check building with submodules
        cfg = Config(
            type='Seq2Pair',
            seqs=Config(type='OTB', version=2015),
            pairs_per_seq=10,
            max_distance=300)
        module = registry.build(cfg)


if __name__ == '__main__':
    unittest.main()
