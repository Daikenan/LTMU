import numpy as np
import time

import neuron.ops as ops
from neuron.models.model import Model


__all__ = ['Tracker', 'OxUvA_Tracker']


class Tracker(Model):

    def __init__(self, name, is_deterministic=True,
                 input_type='image', color_fmt='RGB'):
        assert input_type in ['image', 'file']
        assert color_fmt in ['RGB', 'BGR', 'GRAY']
        super(Tracker, self).__init__()
        self.name = name
        self.is_deterministic = is_deterministic
        self.input_type = input_type
        self.color_fmt = color_fmt
    
    def init(self, img, init_bbox):
        self.init(img, init_bbox)
    
    def update(self, img):
        bboxes = self.update(img)
        return bboxes
    
    def forward_test(self, img_files, init_bbox, visualize=False):
        # state variables
        frame_num = len(img_files)
        bboxes = np.zeros((frame_num, 4))
        bboxes[0] = init_bbox
        times = np.zeros(frame_num)

        for f, img_file in enumerate(img_files):
            if self.input_type == 'image':
                img = ops.read_image(img_file, self.color_fmt)
            elif self.input_type == 'file':
                img = img_file

            begin = time.time()
            if f == 0:
                self.init(img, init_bbox)
            else:
                bboxes[f, :] = self.update(img)
            times[f] = time.time() - begin

            if visualize:
                ops.show_image(img, bboxes[f, :])

        return bboxes, times


class OxUvA_Tracker(Tracker):

    def update(self, img):
        r'''One needs to return (bbox, score, present) in
            function `update`.
        '''
        raise NotImplementedError

    def forward_test(self, img_files, init_bbox, visualize=False):
        frame_num = len(img_files)
        times = np.zeros(frame_num)
        preds = [{
            'present': True,
            'score': 1.0,
            'xmin': init_bbox[0],
            'xmax': init_bbox[2],
            'ymin': init_bbox[1],
            'ymax': init_bbox[3]}]
        
        for f, img_file in enumerate(img_files):
            if self.input_type == 'image':
                img = ops.read_image(img_file, self.color_fmt)
            elif self.input_type == 'file':
                img = img_file

            begin = time.time()
            if f == 0:
                self.init(img, init_bbox)
            else:
                bbox, score, present = self.update(img)
                preds.append({
                    'present': present,
                    'score': score,
                    'xmin': bbox[0],
                    'xmax': bbox[2],
                    'ymin': bbox[1],
                    'ymax': bbox[3]})
            times[f] = time.time() - begin

            if visualize:
                ops.show_image(img, bboxes[f, :])
        
        # update the preds as one-per-second
        frame_stride = 30
        preds = {f * frame_stride: pred for f, pred in enumerate(preds)}

        return preds, times
