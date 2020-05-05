import functools

import tensorflow as tf
from object_detection.trainer import _create_input_queue, _get_inputs
from object_detection.builders import preprocessor_builder, optimizer_builder
from object_detection.utils import variables_helper
import numpy as np
slim=tf.contrib.slim

def _create_losses(input_queue, create_model_fn):
  """Creates loss function for a DetectionModel.

  Args:
    input_queue: BatchQueue object holding enqueued tensor_dicts.
    create_model_fn: A function to create the DetectionModel.
  """
  detection_model = create_model_fn()
  (images, groundtruth_boxes_list, groundtruth_classes_list,
   groundtruth_masks_list
  ) = _get_inputs(input_queue, detection_model.num_classes)
  images = [detection_model.preprocess(image) for image in images]
  images = tf.concat(images, 0)
  if any(mask is None for mask in groundtruth_masks_list):
    groundtruth_masks_list = None

  detection_model.provide_groundtruth(groundtruth_boxes_list,
                                      groundtruth_classes_list,
                                      groundtruth_masks_list)
  prediction_dict = detection_model.predict(images)

  losses_dict = detection_model.loss(prediction_dict)
  for loss_tensor in losses_dict.values():
    tf.losses.add_loss(loss_tensor)
  return images, groundtruth_boxes_list, groundtruth_classes_list



def train(create_model_fn, create_tensor_dict_fn, train_config):
    detection_model = create_model_fn()
    data_augmentation_options = [
              preprocessor_builder.build(step)
              for step in train_config.data_augmentation_options]

    input_queue = _create_input_queue(train_config.batch_size,
                                      create_tensor_dict_fn,
                                      train_config.batch_queue_capacity,
                                      train_config.num_batch_queue_threads,
                                      train_config.prefetch_queue_capacity,
                                      data_augmentation_options)
    images, groundtruth_box_list, groundtruth_class_list = _create_losses(input_queue, create_model_fn)
    groundtruth_box = groundtruth_box_list[0]
    groundtruth_class = groundtruth_class_list[0]

    init_fn = None
    if train_config.fine_tune_checkpoint:
      var_map = detection_model.restore_map(
          from_detection_checkpoint=train_config.from_detection_checkpoint)
      available_var_map = (variables_helper.
                           get_variables_available_in_checkpoint(
                               var_map, train_config.fine_tune_checkpoint))
      init_saver = tf.train.Saver(available_var_map)
      def initializer_fn(sess):
        init_saver.restore(sess, train_config.fine_tune_checkpoint)
      init_fn = initializer_fn
    session_config = tf.ConfigProto(allow_soft_placement=True,
                                    log_device_placement=False)
    sess = tf.Session(config=session_config)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    global_summaries = set([])
    training_optimizer = optimizer_builder.build(train_config.optimizer,
                                                 global_summaries)
    loss = tf.add_n(tf.get_collection(tf.GraphKeys.LOSSES))
    optimize_op = training_optimizer.minimize(loss, global_step=slim.get_global_step())
    sess.run(tf.global_variables_initializer())
    init_fn(sess)
    saver = tf.train.Saver()
    saver.restore(sess, 'model/ssd_mobilenet_mom_imagenet/model.ckpt-49188')
    for i in range(train_config.num_steps):
        img, gt_box, gt_cls, cur_loss = sess.run([images, groundtruth_box, groundtruth_class, loss])
        if not np.isfinite(cur_loss):
            print('Loss is nan')

        #cur_loss, _ = sess.run([loss, optimize_op])
        print("Iteration %d, Loss: %f"%(i, cur_loss))
