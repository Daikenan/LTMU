from object_detection.models.ssd_mobilenet_v1_feature_extractor import SSDMobileNetV1FeatureExtractor
from object_detection.models import feature_map_generators
from nets import mobilenet_v1
import tensorflow as tf
import tensorflow.contrib.slim as slim
import collections

class MobileNetFeaturePyramidExtractor(SSDMobileNetV1FeatureExtractor):
    def extract_features(self, preprocessed_inputs, init_extraction=False):
        """Extract features from preprocessed inputs.
  
        Args:
          preprocessed_inputs: a [batch, height, width, channels] float tensor
            representing a batch of images.
  
        Returns:
          feature_maps: a list of tensors where the ith tensor has shape
            [batch, height_i, width_i, depth_i]
        """
        if init_extraction:
            preprocessed_inputs.get_shape().assert_has_rank(4)
            shape_assert = tf.Assert(
                tf.logical_and(tf.greater_equal(tf.shape(preprocessed_inputs)[1], 33),
                               tf.greater_equal(tf.shape(preprocessed_inputs)[2], 33)),
                ['image size must at least be 33 in both height and width.'])
            with tf.control_dependencies([shape_assert]):
                with slim.arg_scope(self._conv_hyperparams):
                    with tf.variable_scope('MobilenetV1',
                                           reuse=self._reuse_weights) as scope:
                        _, image_features = mobilenet_v1.mobilenet_v1_base(
                            preprocessed_inputs,
                            final_endpoint='Conv2d_13_pointwise',
                            min_depth=self._min_depth,
                            depth_multiplier=self._depth_multiplier,
                            scope=scope)
                        feature_head = image_features['Conv2d_13_pointwise']
                        feature_head = slim.conv2d(
                            feature_head,
                            512, [3,3],
                            stride=1,
                            padding='SAME',
                            scope='Conv2d_Append_1x1_256'
                        )
                        feature_head = tf.nn.avg_pool(feature_head, strides=[1,1,1,1], ksize=[1,4,4,1],
                                                      padding='VALID', )
                        return feature_head
        else:
            preprocessed_inputs.get_shape().assert_has_rank(4)
            shape_assert = tf.Assert(
                tf.logical_and(tf.greater_equal(tf.shape(preprocessed_inputs)[1], 33),
                               tf.greater_equal(tf.shape(preprocessed_inputs)[2], 33)),
                ['image size must at least be 33 in both height and width.'])


            bottomup_features_names = [ 'Conv2d_11_pointwise', 'Conv2d_13_pointwise']
            num_appended_layers = 0
            #appended_channel_num = [512, 256, 256, 256]
            appended_channel_num = [512]

            with tf.control_dependencies([shape_assert]):
                with slim.arg_scope(self._conv_hyperparams):
                    with tf.variable_scope('MobilenetV1',
                                           reuse=self._reuse_weights) as scope:
                        _, image_features = mobilenet_v1.mobilenet_v1_base(
                            preprocessed_inputs,
                            final_endpoint='Conv2d_13_pointwise',
                            min_depth=self._min_depth,
                            depth_multiplier=self._depth_multiplier,
                            scope=scope)

                        topdown_features = self._topdown_feature_maps(
                            image_features,
                            bottomup_features_names=bottomup_features_names,
                            num_appended_layers = num_appended_layers,
                            appended_channel_num = appended_channel_num)
            return topdown_features.values()
    def _topdown_feature_maps(self, image_features, bottomup_features_names,num_appended_layers=2,
                              appended_channel_num=256, stride=2, topdown_channel_num = 512):
        """ Building a top down feature pyramid.
        Args:
            image_features: a dictionary of input bottom_up features with layer names being the keys
            bottomup_features_names: a list of names of selected bottom_up features, which are combined 
                with top down features through a lateral connection. The names are sorted from bottom 
                layers to top layers.
            num_appended_layers: number of layers which are appended to the last bottom up features. 
                Each of the appended layers consists of a 3x3 conv2d followed by a batch_norm and a relus.
                Together with the selected bottom up features, they construct base features of top down branch.
            appended_channel_num: number of channels of output features in appended layers. Could be a scalar or 
                a list of length num_appended_layers.
            stride: stride of the appended layers with respect to the input features.
            topdown_channel_num: number of channels of the output features in the top down branch. Since topdown 
                feature pyramid has the same channel number. This should be a scalar. Topdown layers are firstly 
                resized with nearest neighbor method to have the same with the lateral features and then combined 
                with them through element-wise addition. The lateral features are obtained by applying 1x1 conv2d
                with no nonlinearity to the corresponding bottom up features
            
        Returns:
            topdown_features: An ordered dictionary of the top down feature pyramid.
        """
        # if isinstance(appended_channel_num, list) and len(appended_channel_num) != num_appended_layers:
        #     raise RuntimeError('appened_channel_num should have the length of num_appended_layers')

        # append layers
        feature_head = image_features[bottomup_features_names[-1]]
        appended_features = dict()
        appended_features_names = list()
        for index in range(num_appended_layers):
            if isinstance(appended_channel_num, list):
                num_channel = appended_channel_num[index]
            else:
                num_channel = appended_channel_num
            layer_name = 'Append_{}_Conv2d_3x3_{}'.format(index, num_channel)
            feature_head = slim.conv2d(
                feature_head,
                num_channel, [3,3],
                stride=stride,
                padding='SAME',
                scope=layer_name
            )
            appended_features[layer_name] = feature_head
            appended_features_names.append(layer_name)
        # top down branch
        bottomup_features_names += appended_features_names
        image_features.update(appended_features)
        topdown_features = list()
        topdown_features_names = list()
        # init top_down feature
        level_ind = len(bottomup_features_names)-1
        layer_name = 'TopDown_{}_Conv2d_3x3_{}'.format(level_ind, topdown_channel_num)
        feature_head = slim.conv2d(
            feature_head,
            topdown_channel_num, [3, 3],
            stride=1,
            padding='SAME',
            scope=layer_name
        )
        topdown_features.append(feature_head)
        topdown_features_names.append(layer_name)
        level_ind -= 1
        for bottomup_feature_name in bottomup_features_names[-2::-1]:
            layer_name = 'Lateral_{}_Conv2d_1x1_{}'.format(level_ind, topdown_channel_num)
            lateral_feature = slim.conv2d(
                image_features[bottomup_feature_name],
                topdown_channel_num, [1, 1],
                padding='SAME',
                scope=layer_name)
            output_size = lateral_feature.get_shape().as_list()[1:3]
            if output_size[0] != feature_head.get_shape().as_list()[1]:
                feature_head = tf.image.resize_images(feature_head, output_size,
                                                      method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            feature_head = slim.conv2d(
                feature_head,
                topdown_channel_num, [3,3],
                padding='SAME',
                scope='TopDown_{}_Conv2d_3x3_{}'.format(level_ind, topdown_channel_num)
            )
            layer_name = 'TopDown_{}_Add_{}'.format(level_ind, topdown_channel_num)
            feature_head += lateral_feature
            topdown_features.append(feature_head)
            topdown_features_names.append(layer_name)
            level_ind -= 1

        return collections.OrderedDict(
            [(x, y) for (x, y) in zip(topdown_features_names[-1::-1], topdown_features[-1::-1])])


class MobileNetBoxFeatureExtractor(SSDMobileNetV1FeatureExtractor):
    def extract_features(self, preprocessed_inputs):
        """Extract features from preprocessed inputs.

        Args:
          preprocessed_inputs: a [batch, height, width, channels] float tensor
            representing a batch of images.

        Returns:
          feature_maps: a list of tensors where the ith tensor has shape
            [batch, height_i, width_i, depth_i]
        """
        preprocessed_inputs.get_shape().assert_has_rank(4)
        shape_assert = tf.Assert(
            tf.logical_and(tf.greater_equal(tf.shape(preprocessed_inputs)[1], 33),
                           tf.greater_equal(tf.shape(preprocessed_inputs)[2], 33)),
            ['image size must at least be 33 in both height and width.'])

        bottomup_features_names = ['Conv2d_11_pointwise', 'Conv2d_13_pointwise']
        num_appended_layers = 4
        appended_channel_num = [512, 256, 256, 256]

        with tf.control_dependencies([shape_assert]):
            with slim.arg_scope(self._conv_hyperparams):
                with tf.variable_scope('MobilenetV1',
                                       reuse=self._reuse_weights) as scope:
                    _, image_features = mobilenet_v1.mobilenet_v1_base(
                        preprocessed_inputs,
                        final_endpoint='Conv2d_13_pointwise',
                        min_depth=self._min_depth,
                        depth_multiplier=self._depth_multiplier,
                        scope=scope)
                    topdown_features = self._topdown_feature_maps(
                        image_features,
                        bottomup_features_names=bottomup_features_names,
                        num_appended_layers=num_appended_layers,
                        appended_channel_num=appended_channel_num)
        return topdown_features.values()

    def _topdown_feature_maps(self, image_features, bottomup_features_names, num_appended_layers=2,
                              appended_channel_num=256, stride=2, topdown_channel_num=256):
        """ Building a top down feature pyramid.
        Args:
            image_features: a dictionary of input bottom_up features with layer names being the keys
            bottomup_features_names: a list of names of selected bottom_up features, which are combined 
                with top down features through a lateral connection. The names are sorted from bottom 
                layers to top layers.
            num_appended_layers: number of layers which are appended to the last bottom up features. 
                Each of the appended layers consists of a 3x3 conv2d followed by a batch_norm and a relus.
                Together with the selected bottom up features, they construct base features of top down branch.
            appended_channel_num: number of channels of output features in appended layers. Could be a scalar or 
                a list of length num_appended_layers.
            stride: stride of the appended layers with respect to the input features.
            topdown_channel_num: number of channels of the output features in the top down branch. Since topdown 
                feature pyramid has the same channel number. This should be a scalar. Topdown layers are firstly 
                resized with nearest neighbor method to have the same with the lateral features and then combined 
                with them through element-wise addition. The lateral features are obtained by applying 1x1 conv2d
                with no nonlinearity to the corresponding bottom up features

        Returns:
            topdown_features: An ordered dictionary of the top down feature pyramid.
        """
        if isinstance(appended_channel_num, list) and len(appended_channel_num) != num_appended_layers:
            raise RuntimeError('appened_channel_num should have the length of num_appended_layers')

        # append layers
        feature_head = image_features[bottomup_features_names[-1]]
        appended_features = dict()
        appended_features_names = list()
        for index in range(num_appended_layers):
            if isinstance(appended_channel_num, list):
                num_channel = appended_channel_num[index]
            else:
                num_channel = appended_channel_num
            layer_name = 'Append_{}_Conv2d_3x3_{}'.format(index, num_channel)
            feature_head = slim.conv2d(
                feature_head,
                num_channel, [3, 3],
                stride=stride,
                padding='SAME',
                scope=layer_name
            )
            appended_features[layer_name] = feature_head
            appended_features_names.append(layer_name)
        # top down branch
        bottomup_features_names += appended_features_names
        image_features.update(appended_features)
        topdown_features = list()
        topdown_features_names = list()
        # init top_down feature
        level_ind = len(bottomup_features_names) - 1
        layer_name = 'TopDown_{}_Conv2d_3x3_{}'.format(level_ind, topdown_channel_num)
        feature_head = slim.conv2d(
            feature_head,
            topdown_channel_num, [3, 3],
            stride=1,
            padding='SAME',
            scope=layer_name
        )
        topdown_features.append(feature_head)
        topdown_features_names.append(layer_name)
        level_ind -= 1
        for bottomup_feature_name in bottomup_features_names[-2::-1]:
            layer_name = 'Lateral_{}_Conv2d_1x1_{}'.format(level_ind, topdown_channel_num)
            lateral_feature = slim.conv2d(
                image_features[bottomup_feature_name],
                topdown_channel_num, [1, 1],
                padding='SAME',
                scope=layer_name)
            output_size = lateral_feature.get_shape().as_list()[1:3]
            if output_size[0] != feature_head.get_shape().as_list()[1]:
                feature_head = tf.image.resize_images(feature_head, output_size,
                                                      method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            feature_head = slim.conv2d(
                feature_head,
                topdown_channel_num, [3, 3],
                padding='SAME',
                scope='TopDown_{}_Conv2d_3x3_{}'.format(level_ind, topdown_channel_num)
            )
            layer_name = 'TopDown_{}_Add_{}'.format(level_ind, topdown_channel_num)
            feature_head += lateral_feature
            topdown_features.append(feature_head)
            topdown_features_names.append(layer_name)
            level_ind -= 1

        return collections.OrderedDict(
            [(x, y) for (x, y) in zip(topdown_features_names[-1::-1], topdown_features[-1::-1])])
