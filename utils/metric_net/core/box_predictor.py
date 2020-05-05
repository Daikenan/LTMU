import tensorflow as tf
from object_detection.core import box_predictor
from tensorflow.contrib import slim
from object_detection.utils import shape_utils
BOX_ENCODINGS = 'box_encodings'
CLASS_PREDICTIONS_WITH_BACKGROUND = 'class_predictions_with_background'

class SharedConvolutionalBoxPredictor(box_predictor.BoxPredictor):
    def __init__(self, is_training, num_classes, conv_hyperparams, use_dropout, dropout_keep_probability):
        super(SharedConvolutionalBoxPredictor,self).__init__(is_training, num_classes)
        self._variable_scopes = list()
        self._feature_level = 0
        self._conv_hyperparams = conv_hyperparams
        self._use_dropout=use_dropout
        self._dropout_keep_probability=dropout_keep_probability

    def _construct_subnet(self, image_features, num_conv3, num_conv1,
                          num_channel, output_channel, scope, activation_fn, dropout=True, dropout_keep_prob=0.9,weights_dict=None,conv3=None):
        with tf.variable_scope(scope) as vscope:
            # if scope in self._variable_scopes:
            #     vscope.reuse_variables()
            # else:
            #     self._variable_scopes.append(scope)
            feature_head = image_features

            for index in range(num_conv1):
                if self._use_dropout:
                    feature_head = slim.dropout(feature_head,
                                                keep_prob=self._dropout_keep_probability,
                                                is_training=self._is_training)
                layer_name = 'Conv1x1_{}_{}'.format(index, num_channel)
                feature_head = slim.conv2d(
                    feature_head, num_channel,
                    [1, 1], scope=layer_name
                )
            if self._use_dropout:
                feature_head = slim.dropout(feature_head,
                                            keep_prob=self._dropout_keep_probability,
                                            is_training=self._is_training)

            for index in range(num_conv3):

                layer_name = 'Conv3x3_{}_{}'.format(index, num_channel)
                feature_head = slim.conv2d(
                    feature_head, num_channel,
                    [3,3], scope=layer_name
                )

                if self._use_dropout:
                    feature_head = slim.dropout(feature_head,
                                               keep_prob=self._dropout_keep_probability,
                                               is_training=self._is_training)
            if conv3 is not None:
                layer_name = 'Conv1x1_OutPut_{}'.format(output_channel)
                feature_head = slim.conv2d(
                    feature_head, output_channel,
                    [1, 1], normalizer_fn=None,
                    activation_fn=activation_fn,
                    scope=layer_name
                )

            else:
                layer_name = 'Conv3x3_OutPut_{}'.format(output_channel)
                feature_head = slim.conv2d(
                    feature_head, output_channel,
                    [3, 3], normalizer_fn=None,
                    activation_fn=activation_fn,
                    scope=layer_name
                )

        return feature_head

    def _predict(self, image_features, num_predictions_per_location,reuse=None, **params):
        """Computes encoded object locations and corresponding confidences.
  
        Args:
          image_features: A float tensor of shape [batch_size, height, width,
            channels] containing features for a batch of images.
          num_predictions_per_location: an integer representing the number of box
            predictions to be made per spatial location in the feature map.
  
        Returns:
          A dictionary containing the following tensors.
            box_encodings: A float tensor of shape [batch_size, num_anchors, 1,
              code_size] representing the location of the objects, where
              num_anchors = feat_height * feat_width * num_predictions_per_location
            class_predictions_with_background: A float tensor of shape
              [batch_size, num_anchors, num_classes + 1] representing the class
              predictions for the proposals.
        """

        with tf.variable_scope('BoxPredictor') as scope:
            if reuse:
                scope.reuse_variables()
            num_class_slots = self.num_classes + 1
            with slim.arg_scope(self._conv_hyperparams):
                if self._feature_level == 0:
                    box_encodings = self._construct_subnet(image_features, 1, 1, 256, #base line ready for this architecture
                                                           num_predictions_per_location * 4,
                                                           'BoxReg_%d'%self._feature_level, activation_fn=None)
                    class_predictions_with_background =\
                        self._construct_subnet(image_features, 1, 1, 256,
                                               num_predictions_per_location * num_class_slots,
                                               'BoxCls_%d'%self._feature_level, activation_fn=None)
                else:
                    box_encodings = self._construct_subnet(image_features, 0, 2, 256,
                                                           num_predictions_per_location * 4,
                                                           'BoxReg_%d'%self._feature_level, activation_fn=None,conv3=True)
                    class_predictions_with_background =\
                        self._construct_subnet(image_features, 0, 2, 256,
                                               num_predictions_per_location * num_class_slots,
                                               'BoxCls_%d'%self._feature_level, activation_fn=None,conv3=True)
            self._feature_level += 1
            if self._feature_level == 2:
                self._feature_level = 0

        combined_feature_map_shape = shape_utils.combined_static_and_dynamic_shape(
            image_features)
        box_encodings = tf.reshape(
            box_encodings, tf.stack([combined_feature_map_shape[0],
                                     combined_feature_map_shape[1] *
                                     combined_feature_map_shape[2] *
                                     num_predictions_per_location,
                                     1, 4]))
        class_predictions_with_background = tf.reshape(
            class_predictions_with_background,
            tf.stack([combined_feature_map_shape[0],
                      combined_feature_map_shape[1] *
                      combined_feature_map_shape[2] *
                      num_predictions_per_location,
                      num_class_slots]))
        return {BOX_ENCODINGS: box_encodings,
                CLASS_PREDICTIONS_WITH_BACKGROUND:
                    class_predictions_with_background}
