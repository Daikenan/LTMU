import tensorflow as tf
from object_detection.core import standard_fields as fields
from object_detection.utils import ops as util_ops


def read_video_image(video_path, image_path):
    video_tensor_dict = read_seq(video_path)
    image_tensor_dict = read(image_path)
    return video_tensor_dict, image_tensor_dict



def read(input_config):
    try:
        record_path = input_config.tf_record_input_reader.input_path
    except:
        record_path = input_config
    input_record_queue = tf.train.string_input_producer([record_path])
    record_reader = tf.TFRecordReader()
    _, serialized_example = record_reader.read(input_record_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image_name': tf.FixedLenFeature([1,1], tf.string),
            'bndbox/xmin': tf.FixedLenFeature([], tf.float32),
            'bndbox/ymin': tf.FixedLenFeature([], tf.float32),
            'bndbox/xmax': tf.FixedLenFeature([], tf.float32),
            'bndbox/ymax': tf.FixedLenFeature([], tf.float32)})
    tensor_dict = dict()
    tensor_dict[fields.InputDataFields.filename] = features['image_name']
    bndbox = tf.stack([features['bndbox/ymin'], features['bndbox/xmin'],
                       features['bndbox/ymax'], features['bndbox/xmax']], axis=0)
    bndbox = tf.expand_dims(bndbox, axis=0)
    tensor_dict[fields.InputDataFields.groundtruth_boxes] = bndbox
    classes_gt = tf.ones(shape=[1], dtype=tf.int32)
    label_id_offset = 1
    num_classes = 1
    classes_gt -= label_id_offset
    classes_gt = util_ops.padded_one_hot_encoding(indices=classes_gt,
                                                  depth=num_classes, left_pad=0)
    classes_gt.set_shape([1,1])
    tensor_dict[fields.InputDataFields.groundtruth_classes] = classes_gt
    return tensor_dict


def read_seq(input_config):
    try:
        record_path = input_config.tf_record_input_reader.input_path
    except:
        record_path = input_config
    input_record_queue = tf.train.string_input_producer([record_path])
    record_reader = tf.TFRecordReader()
    _, serialized_example = record_reader.read(input_record_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'folder': tf.FixedLenFeature([], tf.string),
            'image_name': tf.VarLenFeature(tf.string),
            'bndbox/xmin': tf.VarLenFeature(tf.float32),
            'bndbox/ymin': tf.VarLenFeature(tf.float32),
            'bndbox/xmax': tf.VarLenFeature(tf.float32),
            'bndbox/ymax': tf.VarLenFeature(tf.float32)})
    tensor_dict = dict()
    tensor_dict['folder'] = features['folder']

    tensor_dict[fields.InputDataFields.filename] = features['image_name'].values
    bndbox = tf.stack([features['bndbox/ymin'].values, features['bndbox/xmin'].values,
                       features['bndbox/ymax'].values, features['bndbox/xmax'].values], axis=1)
    # bndbox = tf.expand_dims(bndbox, axis=0)

    tensor_dict[fields.InputDataFields.groundtruth_boxes] = bndbox
    classes_gt = tf.ones_like(features['bndbox/ymin'].values, dtype=tf.int32)
    label_id_offset = 1
    num_classes = 1
    classes_gt -= label_id_offset
    classes_gt = util_ops.padded_one_hot_encoding(indices=classes_gt,
                                                  depth=num_classes, left_pad=0)
    tensor_dict[fields.InputDataFields.groundtruth_classes] = classes_gt

    return tensor_dict
