
from abc import abstractmethod

import re
import tensorflow as tf
import functools
from core.target_assigner import TargetAssignerExtend
from object_detection.core import box_list
from object_detection.core import box_predictor as bpredictor
from object_detection.core import model, box_list, box_list_ops, preprocessor
from object_detection.core import standard_fields as fields
from object_detection.core import target_assigner
from object_detection.utils import shape_utils
from object_detection.meta_architectures.ssd_meta_arch import SSDMetaArch
import numpy as np
slim = tf.contrib.slim


def match_and_select_feature(groundtruth_boxes, anchors, feature_maps):
    """ Select features on the groundtruth box locations
    
    Args: 
        groundtruth_boxes: a tensor of batch_size x 4
        anchors: generated anchor Box list
        feature_maps: a list of feature pyramid, each element is a 
            tensor of batch_size x height_i x width_i x channel
    
    Returns:
        selected_features: a tensor of batch_size x 1 x 1 x channel

    """
    groundtruth_boxeslists =[box_list.BoxList(tf.expand_dims(box, 0))
                             for box in tf.unstack(groundtruth_boxes, axis=0)]

    feature_maps = [tf.reshape(m, [m.get_shape()[0].value, -1, m.get_shape()[-1].value]) for m in feature_maps]
    feature_maps = tf.unstack(tf.concat(feature_maps, axis=1), axis=0)
    num_anchors_per_location = anchors.get().get_shape()[0].value / feature_maps[0].get_shape()[0].value
    selected_feature = list()
    for groundtruth_boxes, feature_map in zip(groundtruth_boxeslists, feature_maps):
        iou = box_list_ops.iou(groundtruth_boxes, anchors)
        max_ind = tf.argmax(iou, axis=1) / num_anchors_per_location
        selected_feature.append(tf.gather(feature_map, max_ind))
    selected_feature = tf.concat(selected_feature, axis=0)
    selected_feature = tf.expand_dims(tf.expand_dims(selected_feature, axis= 1), axis=1)
    return selected_feature








class MANMetaArch(SSDMetaArch):
    def __init__(self,
                 is_training,
                 anchor_generator,
                 box_predictor,
                 box_coder,
                 feature_extractor,
                 matcher,
                 region_similarity_calculator,
                 image_resizer_fn,
                 non_max_suppression_fn,
                 score_conversion_fn,
                 classification_loss,
                 localization_loss,
                 classification_loss_weight,
                 localization_loss_weight,
                 normalize_loss_by_num_matches,
                 hard_example_miner,
                 add_summaries=True):
        """SSDMetaArch Constructor.
  
        TODO: group NMS parameters + score converter into
        a class and loss parameters into a class and write config protos for
        postprocessing and losses.
  
        Args:
          is_training: A boolean indicating whether the training version of the
            computation graph should be constructed.
          anchor_generator: an anchor_generator.AnchorGenerator object.
          box_predictor: a box_predictor.BoxPredictor object.
          box_coder: a box_coder.BoxCoder object.
          feature_extractor: a SSDFeatureExtractor object.
          matcher: a matcher.Matcher object.
          region_similarity_calculator: a
            region_similarity_calculator.RegionSimilarityCalculator object.
          image_resizer_fn: a callable for image resizing.  This callable always
            takes a rank-3 image tensor (corresponding to a single image) and
            returns a rank-3 image tensor, possibly with new spatial dimensions.
            See builders/image_resizer_builder.py.
          non_max_suppression_fn: batch_multiclass_non_max_suppression
            callable that takes `boxes`, `scores` and optional `clip_window`
            inputs (with all other inputs already set) and returns a dictionary
            hold tensors with keys: `detection_boxes`, `detection_scores`,
            `detection_classes` and `num_detections`. See `post_processing.
            batch_multiclass_non_max_suppression` for the type and shape of these
            tensors.
          score_conversion_fn: callable elementwise nonlinearity (that takes tensors
            as inputs and returns tensors).  This is usually used to convert logits
            to probabilities.
          classification_loss: an object_detection.core.losses.Loss object.
          localization_loss: a object_detection.core.losses.Loss object.
          classification_loss_weight: float
          localization_loss_weight: float
          normalize_loss_by_num_matches: boolean
          hard_example_miner: a losses.HardExampleMiner object (can be None)
          add_summaries: boolean (default: True) controlling whether summary ops
            should be added to tensorflow graph.
        """
        super(SSDMetaArch, self).__init__(num_classes=box_predictor.num_classes)
        self._is_training = is_training

        # Needed for fine-tuning from classification checkpoints whose
        # variables do not have the feature extractor scope.
        self._extract_features_scope = 'FeatureExtractor'

        self._anchor_generator = anchor_generator
        self._box_predictor = box_predictor

        self._box_coder = box_coder
        self._feature_extractor = feature_extractor
        self._matcher = matcher
        self._region_similarity_calculator = region_similarity_calculator

        # TODO: handle agnostic mode and positive/negative class weights
        unmatched_cls_target = None
        unmatched_cls_target = tf.constant([1] + self.num_classes * [0], tf.float32)
        self._target_assigner = TargetAssignerExtend(
            self._region_similarity_calculator,
            self._matcher,
            self._box_coder,
            positive_class_weight=1.0,
            negative_class_weight=1.0,
            unmatched_cls_target=unmatched_cls_target)

        self._classification_loss = classification_loss
        self._localization_loss = localization_loss
        self._classification_loss_weight = classification_loss_weight
        self._localization_loss_weight = localization_loss_weight
        self._normalize_loss_by_num_matches = normalize_loss_by_num_matches
        self._hard_example_miner = hard_example_miner

        self._image_resizer_fn = image_resizer_fn
        self._non_max_suppression_fn = non_max_suppression_fn
        self._score_conversion_fn = score_conversion_fn

        self._anchors = None
        self._add_summaries = add_summaries



    def _add_box_predictions_to_feature_maps(self, feature_maps,reuse=None):
        """Adds box predictors to each feature map and returns concatenated results.

        Args:
          feature_maps: a list of tensors where the ith tensor has shape
            [batch, height_i, width_i, depth_i]

        Returns:
          box_encodings: 4-D float tensor of shape [batch_size, num_anchors,
              box_code_dimension] containing predicted boxes.
          class_predictions_with_background: 2-D float tensor of shape
              [batch_size, num_anchors, num_classes+1] containing class predictions
              (logits) for each of the anchors.  Note that this tensor *includes*
              background class predictions (at class index 0).

        Raises:
          RuntimeError: if the number of feature maps extracted via the
            extract_features method does not match the length of the
            num_anchors_per_locations list that was passed to the constructor.
          RuntimeError: if box_encodings from the box_predictor does not have
            shape of the form  [batch_size, num_anchors, 1, code_size].
        """
        num_anchors_per_location_list = (
            self._anchor_generator.num_anchors_per_location())
        if len(feature_maps) != len(num_anchors_per_location_list):
            raise RuntimeError('the number of feature maps must match the '
                               'length of self.anchors.NumAnchorsPerLocation().')
        box_encodings_list = []
        cls_predictions_with_background_list = []
        for idx, (feature_map, num_anchors_per_location
                  ) in enumerate(zip(feature_maps, num_anchors_per_location_list)):
            box_predictor_scope = 'BoxPredictor'
            box_predictions = self._box_predictor.predict(feature_map,
                                                          num_anchors_per_location,
                                                          box_predictor_scope,reuse=reuse)
            box_encodings = box_predictions[bpredictor.BOX_ENCODINGS]
            cls_predictions_with_background = box_predictions[
                bpredictor.CLASS_PREDICTIONS_WITH_BACKGROUND]

            box_encodings_shape = box_encodings.get_shape().as_list()
            if len(box_encodings_shape) != 4 or box_encodings_shape[2] != 1:
                raise RuntimeError('box_encodings from the box_predictor must be of '
                                   'shape `[batch_size, num_anchors, 1, code_size]`; '
                                   'actual shape', box_encodings_shape)
            box_encodings = tf.squeeze(box_encodings, axis=2)
            box_encodings_list.append(box_encodings)
            cls_predictions_with_background_list.append(
                cls_predictions_with_background)

        num_predictions = sum(
            [tf.shape(box_encodings)[1] for box_encodings in box_encodings_list])
        num_anchors = self.anchors.num_boxes()
        anchors_assert = tf.assert_equal(num_anchors, num_predictions, [
            'Mismatch: number of anchors vs number of predictions', num_anchors,
            num_predictions
        ])
        with tf.control_dependencies([anchors_assert]):
            box_encodings = tf.concat(box_encodings_list, 1)
            class_predictions_with_background = tf.concat(
                cls_predictions_with_background_list, 1)
        return box_encodings, class_predictions_with_background



    def _add_sequential_box_predictions_to_feature_maps(self, init_feature_maps, feature_maps,reuse=None):
        """Adds box predictors to each feature map and returns concatenated results.
 
        Args:
          feature_maps: a list of tensors where the ith tensor has shape
            [batch, seq_length, height_i, width_i, depth_i]
 
        Returns:
          box_encodings: 4-D float tensor of shape [batch_size, num_anchors,
              box_code_dimension] containing predicted boxes.
          class_predictions_with_background: 2-D float tensor of shape
              [batch_size, num_anchors, num_classes+1] containing class predictions
              (logits) for each of the anchors.  Note that this tensor *includes*
              background class predictions (at class index 0).
 
        Raises:
          RuntimeError: if the number of feature maps extracted via the
            extract_features method does not match the length of the
            num_anchors_per_locations list that was passed to the constructor.
          RuntimeError: if box_encodings from the box_predictor does not have
            shape of the form  [batch_size, num_anchors, 1, code_size].
        """
        # reshape and transpose each feature map to the size of [seq_length, batch_size, height, width, channel]
        feature_maps = [tf.transpose(tf.reshape(m,
                                   [self._batch_size,
                                    self._seq_length,
                                    m.get_shape()[1].value,
                                    m.get_shape()[2].value,
                                    m.get_shape()[3].value]), perm=[1,0,2,3,4]) for m in feature_maps]
        feature_array_list = list()
        for feature_map in feature_maps:
            feature_array = tf.TensorArray(dtype=tf.float32, size=self._seq_length, clear_after_read=False)
            feature_array = feature_array.unstack(feature_map)
            feature_array_list.append(feature_array)
        groundtruth_boxes = tf.transpose(self.groundtruth_lists('boxes'), [1, 0, 2]) # seq_length x batch_size x 4
        # init_groundtruth_boxes = groundtruth_boxes[0]
        box_reg_array = tf.TensorArray(dtype=tf.float32, size=self._seq_length)
        box_cls_array = tf.TensorArray(dtype=tf.float32, size=self._seq_length)
        # groundtruth_boxes_array=groundtruth_boxes_array.unstack(groundtruth_boxes[1:])
        # selected_feature = init_feature_list[-1]
        # selected_feature = match_and_select_feature(init_groundtruth_boxes, self.anchors, init_feature_list)

        # feature_maps = [A.read(0) for A in feature_array_list]
        # feature_maps = [tf.concat([m,
        #                            tf.tile(selected_feature, [1, m.get_shape()[1].value, m.get_shape()[2].value, 1])],
        #                           axis=3) for m in feature_maps]
        # a = self._add_box_predictions_to_feature_maps(feature_maps)

        def _time_step(time, box_reg_array, box_cls_array):
            feature_maps = [A.read(time) for A in feature_array_list]
            concate_feature_maps = list()
            for m in feature_maps:
                #tile_num = int(np.ceil(int(m.get_shape()[1])/ float(int(init_feature_maps.get_shape()[1]))))
                tile_num = int(m.get_shape()[1])
                tiled_init_feature = tf.tile(init_feature_maps,
                                             [1,tile_num,tile_num,1])
                #crop_shape = int(m.get_shape()[1])
                #tiled_init_feature = tiled_init_feature[:,:crop_shape,:crop_shape,:]
                tmp1 = m*tiled_init_feature
                concate_feature_maps.append(tf.concat([tf.nn.l2_normalize(tmp1,dim=3)*10, tf.nn.l2_normalize(tiled_init_feature,dim=3)*10], axis=3))
            reg_pre, cls_pre = self._add_box_predictions_to_feature_maps(concate_feature_maps,reuse=reuse)
            box_reg_array = box_reg_array.write(time, reg_pre)
            box_cls_array = box_cls_array.write(time, cls_pre)
            # groundtruth_boxes = groundtruth_boxes_array.read(time)
            # selected_feature = match_and_select_feature(groundtruth_boxes, self.anchors, feature_maps)
            return time+1, box_reg_array, box_cls_array
        _, box_reg_array, box_cls_array = tf.while_loop(
            cond=lambda time, *_: time < self._seq_length,
            body=_time_step,
            loop_vars=(0,  box_reg_array, box_cls_array),
            parallel_iterations=1,
            swap_memory=True)
        #_, _, box_reg_array, box_cls_array = _time_step(0, selected_feature, box_reg_array, box_cls_array)
        box_reg = box_reg_array.stack()
        box_reg.set_shape([self._seq_length,
                           self._batch_size,
                           box_reg.get_shape()[2].value,
                           box_reg.get_shape()[3].value])
        box_reg = tf.transpose(box_reg, [1,0,2,3])
        box_reg = tf.reshape(box_reg, [self._batch_size * (self._seq_length),
                                       box_reg.get_shape()[2].value,
                                       box_reg.get_shape()[3].value])
        box_cls = box_cls_array.stack()
        box_cls.set_shape([self._seq_length,
                           self._batch_size,
                           box_cls.get_shape()[2].value,
                           box_cls.get_shape()[3].value])
        box_cls = tf.transpose(box_cls, [1,0,2,3])
        box_cls = tf.reshape(box_cls, [self._batch_size * (self._seq_length),
                                       box_cls.get_shape()[2].value,
                                       box_cls.get_shape()[3].value])
        return box_reg, box_cls

    def _assign_targets(self, groundtruth_boxes_list, groundtruth_classes_list):
        """Assign groundtruth targets.
  
        Adds a background class to each one-hot encoding of groundtruth classes
        and uses target assigner to obtain regression and classification targets.
  
        Args:
          groundtruth_boxes_list: a list of 2-D tensors of shape [num_boxes, 4]
            containing coordinates of the groundtruth boxes.
              Groundtruth boxes are provided in [y_min, x_min, y_max, x_max]
              format and assumed to be normalized and clipped
              relative to the image window with y_min <= y_max and x_min <= x_max.
          groundtruth_classes_list: a list of 2-D one-hot (or k-hot) tensors of
            shape [num_boxes, num_classes] containing the class targets with the 0th
            index assumed to map to the first non-background class.
  
        Returns:
          batch_cls_targets: a tensor with shape [batch_size, num_anchors,
            num_classes],
          batch_cls_weights: a tensor with shape [batch_size, num_anchors],
          batch_reg_targets: a tensor with shape [batch_size, num_anchors,
            box_code_dimension]
          batch_reg_weights: a tensor with shape [batch_size, num_anchors],
          match_list: a list of matcher.Match objects encoding the match between
            anchors and groundtruth boxes for each image of the batch,
            with rows of the Match objects corresponding to groundtruth boxes
            and columns corresponding to anchors.
        """

        groundtruth_boxes_list = tf.reshape(groundtruth_boxes_list, [self._batch_size*(self._seq_length), -1])
        groundtruth_boxes_list = tf.unstack(groundtruth_boxes_list, axis=0)
        groundtruth_boxlists = [
            box_list.BoxList(tf.expand_dims(boxes, axis=0)) for boxes in groundtruth_boxes_list
        ]

        groundtruth_classes_list = tf.reshape(groundtruth_classes_list, [self._batch_size*(self._seq_length), -1])
        groundtruth_classes_list = tf.unstack(groundtruth_classes_list, axis=0)
        groundtruth_classes_with_background_list = [
            tf.reshape(tf.one_hot(one_hot_encoding, self.num_classes+1), [1, self.num_classes+1])
            for one_hot_encoding in groundtruth_classes_list
        ]

        return target_assigner.batch_assign_targets(
            self._target_assigner, self.anchors, groundtruth_boxlists,
            groundtruth_classes_with_background_list)

    def extract_feature(self,preprocessed_inputs):
        self._batch_size, self._seq_length, self._input_size, _, _ = preprocessed_inputs.get_shape().as_list()

        preprocessed_inputs = tf.reshape(preprocessed_inputs, [-1, self._input_size, self._input_size, 3])
        with tf.variable_scope('FeatureExtractor'):
            feature_maps = self._feature_extractor.extract_features(
                preprocessed_inputs)

        output_dict = {
            'feature_maps0': feature_maps[0],
            'feature_maps1': feature_maps[1],
            # 'feature_maps2': feature_maps[2],
            # 'feature_maps3': feature_maps[3],
            # 'feature_maps4': feature_maps[4],
            # 'feature_maps5': feature_maps[5]
        }

        return output_dict

    def extract_init_feature(self,preprocessed_init_input):
        _, _, self.init_input_size, _, _ = preprocessed_init_input.get_shape().as_list()
        preprocessed_init_input = tf.reshape(preprocessed_init_input,
                                             [-1, self.init_input_size, self.init_input_size, 3])

        with tf.variable_scope('InitFeatureExtractor'):
            init_feature_maps = self._feature_extractor.extract_features(
                preprocessed_init_input, True)

        return init_feature_maps

    def predict_box_with_init(self,init_feature_maps, preprocessed_input,istraining=False):

        self._batch_size, self._seq_length, self._input_size, _, _ = preprocessed_input.get_shape().as_list()
        preprocessed_inputs = tf.reshape(preprocessed_input, [-1, self._input_size, self._input_size, 3])


        with tf.variable_scope('FeatureExtractor'):
            feature_maps = self._feature_extractor.extract_features(
                preprocessed_inputs)

        self._is_training = istraining
        feature_map_spatial_dims = self._get_feature_map_spatial_dims(feature_maps)
        self._anchors = self._anchor_generator.generate(feature_map_spatial_dims)
        (box_encodings, class_predictions_with_background
         ) = self._add_sequential_box_predictions_to_feature_maps(init_feature_maps, feature_maps)
        predictions_dict = {
            'box_encodings': box_encodings,
            'class_predictions_with_background': class_predictions_with_background,
            'feature_maps': feature_maps
        }

        return predictions_dict


    def predict_box(self,init_feature_maps, feature_maps,istraining=False):

        self._is_training = istraining
        feature_map_spatial_dims = self._get_feature_map_spatial_dims(feature_maps)
        self._anchors = self._anchor_generator.generate(feature_map_spatial_dims)
        (box_encodings, class_predictions_with_background
         ) = self._add_sequential_box_predictions_to_feature_maps(init_feature_maps, feature_maps)
        predictions_dict = {
            'box_encodings': box_encodings,
            'class_predictions_with_background': class_predictions_with_background,
            'feature_maps': feature_maps
        }

        return predictions_dict

    def predict(self, preprocessed_init_input, preprocessed_inputs,istraining=False,reuse=None,weights_dict=None):
        """Predicts unpostprocessed tensors from input tensor.
  
        This function takes an input batch of images and runs it through the forward
        pass of the network to yield unpostprocessesed predictions.
  
        A side effect of calling the predict method is that self._anchors is
        populated with a box_list.BoxList of anchors.  These anchors must be
        constructed before the postprocess or loss functions can be called.
  
        Args:
          preprocessed_inputs: a [batch, height, width, channels] image tensor.
  
        Returns:
          prediction_dict: a dictionary holding "raw" prediction tensors:
            1) box_encodings: 4-D float tensor of shape [batch_size, num_anchors,
              box_code_dimension] containing predicted boxes.
            2) class_predictions_with_background: 3-D float tensor of shape
              [batch_size, num_anchors, num_classes+1] containing class predictions
              (logits) for each of the anchors.  Note that this tensor *includes*
              background class predictions (at class index 0).
            3) feature_maps: a list of tensors where the ith tensor has shape
              [batch, height_i, width_i, depth_i].
        """
        self._is_training = istraining
        self._batch_size, self._seq_length, self._input_size, _, _ = preprocessed_inputs.get_shape().as_list()
        _, _, self.init_input_size, _, _ = preprocessed_init_input.get_shape().as_list()
        preprocessed_init_input = tf.reshape(preprocessed_init_input,
                                             [-1, self.init_input_size, self.init_input_size, 3])
        preprocessed_inputs = tf.reshape(preprocessed_inputs, [-1, self._input_size, self._input_size, 3])
        with tf.variable_scope('InitFeatureExtractor') as scope:
            if reuse:
                scope.reuse_variables()
            init_feature_maps = self._feature_extractor.extract_features(
                preprocessed_init_input, True)
        with tf.variable_scope('FeatureExtractor') as scope:
            if reuse:
                scope.reuse_variables()
            feature_maps = self._feature_extractor.extract_features(
                preprocessed_inputs)
        # if weights_dict is None:
        #     weights_dict = dict()
        #     weights_dict = self._box_predictor.get_weights(weights_dict,reuse=reuse)

        feature_map_spatial_dims = self._get_feature_map_spatial_dims(feature_maps)
        self._anchors = self._anchor_generator.generate(feature_map_spatial_dims)
        (box_encodings, class_predictions_with_background
         ) = self._add_sequential_box_predictions_to_feature_maps(init_feature_maps, feature_maps,reuse=reuse)
        predictions_dict = {
            'box_encodings': box_encodings,
            'class_predictions_with_background': class_predictions_with_background,
            'feature_maps': feature_maps
        }
        return predictions_dict

    def preprocess(self, inputs, win_size=None):
        """Feature-extractor specific preprocessing.
  
        See base class.
  
        Args:
          inputs: a [batch, height_in, width_in, channels] float tensor representing
            a batch of images with values between 0 and 255.0.
  
        Returns:
          preprocessed_inputs: a [batch, height_out, width_out, channels] float
            tensor representing a batch of images.
        Raises:
          ValueError: if inputs tensor does not have type tf.float32
        """
        if inputs.dtype is not tf.float32:
            raise ValueError('`preprocess` expects a tf.float32 tensor')
        if win_size == None:
            win_size=[300,300]
        with tf.name_scope('Preprocessor'):
            # TODO: revisit whether to always use batch size as  the number of
            # parallel iterations vs allow for dynamic batching.
            _image_resizer_fn = functools.partial(preprocessor.resize_image, new_height=win_size[0], new_width=win_size[1])
            resized_inputs = tf.map_fn(_image_resizer_fn,
                                       elems=inputs,
                                       dtype=tf.float32)
            return self._feature_extractor.preprocess(resized_inputs)

    def restore_map(self, from_detection_checkpoint=True):
        """Returns a map of variables to load from a foreign checkpoint.
  
        See parent class for details.
  
        Args:
          from_detection_checkpoint: whether to restore from a full detection
            checkpoint (with compatible variable names) or to restore from a
            classification checkpoint for initialization prior to training.
  
        Returns:
          A dict mapping variable names (to load from a checkpoint) to variables in
          the model graph.
        """
        variables_to_restore = {}
        for variable in tf.all_variables():
            if variable.op.name.startswith(self._extract_features_scope):
                var_name = variable.op.name
                if not from_detection_checkpoint:
                    var_name = (re.split('^' + self._extract_features_scope + '/',
                                         var_name)[-1])
                variables_to_restore[var_name] = variable
        return variables_to_restore

    def restore_init_map(self, from_detection_checkpoint=True):
        """Returns a map of variables to load from a foreign checkpoint.

        See parent class for details.

        Args:
          from_detection_checkpoint: whether to restore from a full detection
            checkpoint (with compatible variable names) or to restore from a
            classification checkpoint for initialization prior to training.

        Returns:
          A dict mapping variable names (to load from a checkpoint) to variables in
          the model graph.
        """
        variables_to_restore = {}
        for variable in tf.all_variables():
            if variable.op.name.startswith('InitFeatureExtractor'):
                var_name = variable.op.name
                if not from_detection_checkpoint:
                    var_name = (re.split('^' + 'InitFeatureExtractor' + '/',
                                         var_name)[-1])
                else:
                    var_name = (re.split('^' + 'Init',
                                         var_name)[-1])
                variables_to_restore[var_name] = variable
        return variables_to_restore
