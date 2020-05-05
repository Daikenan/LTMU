from object_detection.core.target_assigner import TargetAssigner
import tensorflow as tf
from object_detection.core import box_list

class TargetAssignerExtend(TargetAssigner):
    def assign(self, anchors, groundtruth_boxes, groundtruth_labels=None,
               **params):
        """Assign classification and regression targets to each anchor.
        The extended version assign 0 weights to negative (0) box regression.
  
        For a given set of anchors and groundtruth detections, match anchors
        to groundtruth_boxes and assign classification and regression targets to
        each anchor as well as weights based on the resulting match (specifying,
        e.g., which anchors should not contribute to training loss).
  
        Anchors that are not matched to anything are given a classification target
        of self._unmatched_cls_target which can be specified via the constructor.
        
  
        Args:
          anchors: a BoxList representing N anchors
          groundtruth_boxes: a BoxList representing M groundtruth boxes
          groundtruth_labels:  a tensor of shape [num_gt_boxes, d_1, ... d_k]
            with labels for each of the ground_truth boxes. The subshape
            [d_1, ... d_k] can be empty (corresponding to scalar inputs).  When set
            to None, groundtruth_labels assumes a binary problem where all
            ground_truth boxes get a positive label (of 1).
          **params: Additional keyword arguments for specific implementations of
                  the Matcher.
  
        Returns:
          cls_targets: a float32 tensor with shape [num_anchors, d_1, d_2 ... d_k],
            where the subshape [d_1, ..., d_k] is compatible with groundtruth_labels
            which has shape [num_gt_boxes, d_1, d_2, ... d_k].
          cls_weights: a float32 tensor with shape [num_anchors]
          reg_targets: a float32 tensor with shape [num_anchors, box_code_dimension]
          reg_weights: a float32 tensor with shape [num_anchors]
          match: a matcher.Match object encoding the match between anchors and
            groundtruth boxes, with rows corresponding to groundtruth boxes
            and columns corresponding to anchors.
  
        Raises:
          ValueError: if anchors or groundtruth_boxes are not of type
            box_list.BoxList
        """
        if not isinstance(anchors, box_list.BoxList):
            raise ValueError('anchors must be an BoxList')
        if not isinstance(groundtruth_boxes, box_list.BoxList):
            raise ValueError('groundtruth_boxes must be an BoxList')

        if groundtruth_labels is None:
            groundtruth_labels = tf.ones(tf.expand_dims(groundtruth_boxes.num_boxes(),
                                                        0))
            groundtruth_labels = tf.expand_dims(groundtruth_labels, -1)
        shape_assert = tf.assert_equal(tf.shape(groundtruth_labels)[1:],
                                       tf.shape(self._unmatched_cls_target))

        with tf.control_dependencies([shape_assert]):
            match_quality_matrix = self._similarity_calc.compare(groundtruth_boxes,
                                                                 anchors)
            match = self._matcher.match(match_quality_matrix, **params)
            reg_targets = self._create_regression_targets(anchors,
                                                          groundtruth_boxes,
                                                          match)
            cls_targets = self._create_classification_targets(groundtruth_labels,
                                                              match)
            reg_weights = self._create_regression_weights(match, groundtruth_labels)
            cls_weights = self._create_classification_weights(
                match, self._positive_class_weight, self._negative_class_weight)

            num_anchors = anchors.num_boxes_static()
            if num_anchors is not None:
                reg_targets = self._reset_target_shape(reg_targets, num_anchors)
                cls_targets = self._reset_target_shape(cls_targets, num_anchors)
                reg_weights = self._reset_target_shape(reg_weights, num_anchors)
                cls_weights = self._reset_target_shape(cls_weights, num_anchors)

        return cls_targets, cls_weights, reg_targets, reg_weights, match

    def _create_regression_weights(self, match, groundtruth_labels):
        """Set regression weight for each anchor.
 
        Only positive anchors are set to contribute to the regression loss, so this
        method returns a weight of 1 for every positive anchor and 0 for every
        negative anchor.
 
        Args:
          match: a matcher.Match object that provides a matching between anchors
            and groundtruth boxes.
 
        Returns:
          reg_weights: a float32 tensor with shape [num_anchors] representing
            regression weights
        """

        reg_weights = tf.cast(match.matched_column_indicator(), tf.float32)

        matched_gt_indices = match.matched_row_indices()
        matched_label = tf.gather(groundtruth_labels, matched_gt_indices)
        matched_is_foreground = tf.cast(matched_label[:,0] <= 0, tf.float32)
        matched_anchor_indices = match.matched_column_indices()
        unmatched_ignored_anchor_indices=match.unmatched_or_ignored_column_indices()
        unmatched_ignored_reg_weights = tf.gather(reg_weights, unmatched_ignored_anchor_indices)
        reg_weights= tf.dynamic_stitch(
            [matched_anchor_indices, unmatched_ignored_anchor_indices],
            [matched_is_foreground, unmatched_ignored_reg_weights])
        return reg_weights

