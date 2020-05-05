from object_detection.builders import anchor_generator_builder
from object_detection.builders import box_coder_builder
from core.man_anchor_generator import create_man_anchors
from core import hyperparams_builder
from object_detection.builders import image_resizer_builder
from object_detection.builders import losses_builder
from object_detection.builders import matcher_builder
from object_detection.builders import post_processing_builder
from object_detection.builders import region_similarity_calculator_builder as sim_calc
from object_detection.models.ssd_mobilenet_v1_feature_extractor import SSDMobileNetV1FeatureExtractor
from core.feature_extractor import MobileNetFeaturePyramidExtractor
from object_detection.core import box_predictor
from core.man_meta_arch import MANMetaArch
from core.box_predictor import SharedConvolutionalBoxPredictor
from google.protobuf import text_format
from object_detection.protos import model_pb2
import tensorflow as tf

def _build_man_feature_extractor(feature_extractor_config, is_training,
                                 reuse_weights=None):
    depth_multiplier = feature_extractor_config.depth_multiplier
    min_depth = feature_extractor_config.min_depth
    conv_hyperparams = hyperparams_builder.build(
         feature_extractor_config.conv_hyperparams, is_training)
    return MobileNetFeaturePyramidExtractor(depth_multiplier, min_depth, conv_hyperparams,
                                    reuse_weights)

def _build_man_box_predictor(is_training, num_classes, box_predictor_config):
    conv_box_predictor = box_predictor_config.convolutional_box_predictor
    conv_hyperparams = hyperparams_builder.build(conv_box_predictor.conv_hyperparams,
                                   is_training)

    return SharedConvolutionalBoxPredictor(
        is_training,
        num_classes,
        conv_hyperparams,
        conv_box_predictor.use_dropout,
        conv_box_predictor.dropout_keep_probability)


def _build_man_anchor_generator(anchor_generator_config):
    ssd_anchor_generator_config = anchor_generator_config.ssd_anchor_generator
    return create_man_anchors(
        num_layers=ssd_anchor_generator_config.num_layers,
        min_scale=ssd_anchor_generator_config.min_scale,
        max_scale=ssd_anchor_generator_config.max_scale,
        aspect_ratios=ssd_anchor_generator_config.aspect_ratios,
        reduce_boxes_in_lowest_layer=(ssd_anchor_generator_config
                                      .reduce_boxes_in_lowest_layer))


def build_man_model(model_config, is_training):

    num_classes = model_config.num_classes
    feature_extractor = _build_man_feature_extractor(model_config.feature_extractor,
                                                     is_training)

    box_coder = box_coder_builder.build(model_config.box_coder)
    matcher = matcher_builder.build(model_config.matcher)
    region_similarity_calculator = sim_calc.build(
        model_config.similarity_calculator)
    ssd_box_predictor = _build_man_box_predictor(is_training, num_classes, model_config.box_predictor)
    # ssd_box_predictor = box_predictor_builder.build(hyperparams_builder.build,
    #                                                 model_config.box_predictor,
    #                                                 is_training, num_classes)
    anchor_generator = _build_man_anchor_generator(model_config.anchor_generator)
    # anchor_generator = anchor_generator_builder.build(
    #     model_config.anchor_generator)
    image_resizer_fn = image_resizer_builder.build(model_config.image_resizer)
    non_max_suppression_fn, score_conversion_fn = post_processing_builder.build(
        model_config.post_processing)
    (classification_loss, localization_loss, classification_weight,
     localization_weight,
     hard_example_miner) = losses_builder.build(model_config.loss)
    normalize_loss_by_num_matches = model_config.normalize_loss_by_num_matches

    return MANMetaArch(
        is_training,
        anchor_generator,
        ssd_box_predictor,
        box_coder,
        feature_extractor,
        matcher,
        region_similarity_calculator,
        image_resizer_fn,
        non_max_suppression_fn,
        score_conversion_fn,
        classification_loss,
        localization_loss,
        classification_weight,
        localization_weight,
        normalize_loss_by_num_matches,
        hard_example_miner,
        add_summaries=False)







# fid = open('model.config', 'r')
# model_config = model_pb2.DetectionModel()
# text_format.Merge(fid.read(), model_config)
# fid.close()
# model_config = model_config.ssd
# is_training = True
# detection_model = build_man_model(model_config, is_training)
# 1
