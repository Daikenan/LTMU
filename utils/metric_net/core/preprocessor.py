from object_detection.core.preprocessor import random_crop_image, random_horizontal_flip
from object_detection.core import standard_fields as fields
import tensorflow as tf

def preprocess(tensor_dict):


    images = tensor_dict[fields.InputDataFields.image]
    if len(images.get_shape()) != 4:
        raise ValueError('images in tensor_dict should be rank 4')
    images = tf.squeeze(images, squeeze_dims=[0])
    boxes = tensor_dict[fields.InputDataFields.groundtruth_boxes]
    class_label = tensor_dict[fields.InputDataFields.groundtruth_classes]

    flipped_image, flipped_box = random_horizontal_flip(images, boxes)
    cropped_image, cropped_box, cropped_label = random_crop_image(flipped_image,
                                                                  flipped_box, class_label,
                                                                  aspect_ratio_range=(0.5, 2),
                                                                  area_range=(0.1, 1.0),
                                                                  overlap_thresh=0.3,
                                                                  random_coef=0.15)
    res_tensor_dict = tensor_dict.copy()
    res_tensor_dict[fields.InputDataFields.image] = tf.expand_dims(cropped_image, 0)
    res_tensor_dict[fields.InputDataFields.groundtruth_boxes] = cropped_box
    res_tensor_dict[fields.InputDataFields.groundtruth_classes] = cropped_label

    return res_tensor_dict