from object_detection.anchor_generators.multiple_grid_anchor_generator import MultipleGridAnchorGenerator
import numpy as np
import tensorflow as tf

def create_man_anchors(num_layers=6,
                       min_scale=0.2,
                       max_scale=0.95,
                       aspect_ratios=(1.0, 2.0, 3.0, 1.0/2, 1.0/3),
                       base_anchor_size=None,
                       reduce_boxes_in_lowest_layer=True):
    """Creates MultipleGridAnchorGenerator for SSD anchors.
    
    This function instantiates a MultipleGridAnchorGenerator that reproduces
    ``default box`` construction proposed by Liu et al in the SSD paper.
    See Section 2.2 for details. Grid sizes are assumed to be passed in
    at generation time from finest resolution to coarsest resolution --- this is
    used to (linearly) interpolate scales of anchor boxes corresponding to the
    intermediate grid sizes.
    
    Anchors that are returned by calling the `generate` method on the returned
    MultipleGridAnchorGenerator object are always in normalized coordinates
    and clipped to the unit square: (i.e. all coordinates lie in [0, 1]x[0, 1]).
    
    Args:
    num_layers: integer number of grid layers to create anchors for (actual
      grid sizes passed in at generation time)
    min_scale: scale of anchors corresponding to finest resolution (float)
    max_scale: scale of anchors corresponding to coarsest resolution (float)
    aspect_ratios: list or tuple of (float) aspect ratios to place on each
      grid point.
    base_anchor_size: base anchor size as [height, width].
    reduce_boxes_in_lowest_layer: a boolean to indicate whether the fixed 3
      boxes per location is used in the lowest layer.
    
    Returns:
    a MultipleGridAnchorGenerator
    """
    if base_anchor_size is None:
        base_anchor_size = [1.0, 1.0]
    base_anchor_size = tf.constant(base_anchor_size, dtype=tf.float32)
    box_specs_list = []
    scales = 0.5*2**np.linspace(min_scale, max_scale, num_layers)
    for layer, scale in enumerate(scales):
        layer_box_specs = []
        for aspect_ratio in aspect_ratios:
            if layer < len(scales) - 1:
                for r in np.linspace(0,0.1,2):
                    layer_box_specs.append((scale*2**(r), aspect_ratio))
            else:
                layer_box_specs.append((scale*2**(0), aspect_ratio))
        box_specs_list.append(layer_box_specs)
    return MultipleGridAnchorGenerator(box_specs_list, base_anchor_size)
