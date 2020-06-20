from neuron.config import registry
from .tracker import Tracker, OxUvA_Tracker


__all__ = ['DummyTracker', 'DummyOxUvA_Tracker']


@registry.register_module
class DummyTracker(Tracker):

    def __init__(self):
        super(DummyTracker, self).__init__(
            name='Dummy', is_deterministic=True, input_type='file')
    
    def init(self, img, init_bbox):
        self.bbox = init_bbox
    
    def update(self, img):
        return self.bbox


@registry.register_module
class DummyOxUvA_Tracker(OxUvA_Tracker):

    def __init__(self):
        super(DummyOxUvA_Tracker, self).__init__(
            name='Dummy', is_deterministic=True, input_type='file')
    
    def init(self, img, init_bbox):
        self.bbox = init_bbox
    
    def update(self, img):
        return self.bbox, 1.0, True
