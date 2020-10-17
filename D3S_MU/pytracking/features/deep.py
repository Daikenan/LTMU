from pytracking.features.featurebase import FeatureBase, MultiFeatureBase
import torch
import torchvision
from pytracking import TensorList
from pytracking.evaluation.environment import env_settings
import os
from ltr import load_network
from ltr.models.backbone.resnet18_vggm import resnet18_vggmconv1
import ltr.models.backbone.resnet as backbone_resnet

normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])


class ResNet18m1(MultiFeatureBase):
    """ResNet18 feature together with the VGG-m conv1 layer.
    args:
        output_layers: List of layers to output.
        net_path: Relative or absolute net path (default should be fine).
        use_gpu: Use GPU or CPU.
    """
    def __init__(self,  output_layers, net_path=None, use_gpu=True, *args, **kwargs):
        super(ResNet18m1, self).__init__(*args, **kwargs)

        for l in output_layers:
            if l not in ['vggconv1', 'conv1', 'layer1', 'layer2', 'layer3', 'layer4', 'fc']:
                raise ValueError('Unknown layer')

        self.output_layers = list(output_layers)
        self.use_gpu = use_gpu
        self.net_path = 'resnet18_vggmconv1.pth' if net_path is None else net_path

    def initialize(self):
        if os.path.isabs(self.net_path):
            net_path_full = self.net_path
        else:
            net_path_full = os.path.join(env_settings().network_path, self.net_path)

        if isinstance(self.pool_stride, int) and self.pool_stride == 1:
            self.pool_stride = [1]*len(self.output_layers)

        self.layer_stride = {'vggconv1': 2, 'conv1': 2, 'layer1': 4, 'layer2': 8, 'layer3': 16, 'layer4': 32, 'fc': None}
        self.layer_dim = {'vggconv1': 96, 'conv1': 64, 'layer1': 64, 'layer2': 128, 'layer3': 256, 'layer4': 512, 'fc': None}

        self.mean = torch.Tensor([0.485, 0.456, 0.406]).view(1,-1,1,1)
        self.std = torch.Tensor([0.229, 0.224, 0.225]).view(1,-1,1,1)

        self.net = resnet18_vggmconv1(self.output_layers, path=net_path_full)
        if self.use_gpu:
            self.net.cuda()
        self.net.eval()

    def free_memory(self):
        if hasattr(self, 'net'):
            del self.net

    def dim(self):
        return TensorList([self.layer_dim[l] for l in self.output_layers])

    def stride(self):
        return TensorList([s * self.layer_stride[l] for l, s in zip(self.output_layers, self.pool_stride)])

    def extract(self, im: torch.Tensor):
        im = im/255
        im -= self.mean
        im /= self.std

        if self.use_gpu:
            im = im.cuda()

        with torch.no_grad():
            return TensorList(self.net(im).values())



class ATOMResNet18(MultiFeatureBase):
    """ResNet18 feature with the ATOM IoUNet.
    args:
        output_layers: List of layers to output.
        net_path: Relative or absolute net path (default should be fine).
        use_gpu: Use GPU or CPU.
    """
    def __init__(self, output_layers=('layer3',), net_path='atom_iou', use_gpu=True, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.output_layers = list(output_layers)
        self.use_gpu = use_gpu
        self.net_path = net_path

    def initialize(self):
        # if os.path.isabs(self.net_path):
        #     net_path_full = self.net_path
        # else:
        #     net_path_full = os.path.join(env_settings().network_path, self.net_path)
        #
        # self.net, _ = load_network(net_path_full, backbone_pretrained=False)

        # Modified to perform ablation study using ResNet18
        self.net = backbone_resnet.resnet18(output_layers=self.output_layers, pretrained=True)

        if self.use_gpu:
            self.net.cuda()
        self.net.eval()

        #self.iou_predictor = self.net.bb_regressor

        self.layer_stride = {'conv1': 2, 'layer1': 4, 'layer2': 8, 'layer3': 16, 'layer4': 32, 'classification': 16, 'fc': None}
        self.layer_dim = {'conv1': 64, 'layer1': 64, 'layer2': 128, 'layer3': 256, 'layer4': 512, 'classification': 256,'fc': None}

        #self.iounet_feature_layers = self.net.bb_regressor_layer

        if isinstance(self.pool_stride, int) and self.pool_stride == 1:
            self.pool_stride = [1]*len(self.output_layers)

        #self.feature_layers = sorted(list(set(self.output_layers + self.iounet_feature_layers)))

        self.mean = torch.Tensor([0.485, 0.456, 0.406]).view(1,-1,1,1)
        self.std = torch.Tensor([0.229, 0.224, 0.225]).view(1,-1,1,1)

    def free_memory(self):
        if hasattr(self, 'net'):
            del self.net
        if hasattr(self, 'iou_predictor'):
            del self.iou_predictor
        if hasattr(self, 'iounet_backbone_features'):
            del self.iounet_backbone_features
        if hasattr(self, 'iounet_features'):
            del self.iounet_features

    def dim(self):
        return TensorList([self.layer_dim[l] for l in self.output_layers])

    def stride(self):
        return TensorList([s * self.layer_stride[l] for l, s in zip(self.output_layers, self.pool_stride)])

    def extract(self, im: torch.Tensor):
        im = im/255
        im -= self.mean
        im /= self.std

        if self.use_gpu:
            im = im.cuda()

        with torch.no_grad():
            #output_features = self.net.extract_features(im, self.feature_layers)
            output_features = self.net(im, output_layers=self.output_layers)

        # Store the raw resnet features which are input to iounet
        #self.iounet_backbone_features = TensorList([output_features[layer].clone() for layer in self.iounet_feature_layers])

        # Store the processed features from iounet, just before pooling
        #with torch.no_grad():
        #    self.iounet_features = TensorList(self.iou_predictor.get_iou_feat(self.iounet_backbone_features))

        return TensorList([output_features[layer] for layer in self.output_layers])




class ATOMResNet50(MultiFeatureBase):
    """ResNet18 feature with the ATOM IoUNet.
    args:
        output_layers: List of layers to output.
        net_path: Relative or absolute net path (default should be fine).
        use_gpu: Use GPU or CPU.
    """
    def __init__(self, output_layers=('layer3',), net_path='atom_iou', use_gpu=True, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.output_layers = list(output_layers)
        self.use_gpu = use_gpu
        self.net_path = net_path

    def initialize(self):
        # if os.path.isabs(self.net_path):
        #     net_path_full = self.net_path
        # else:
        #     net_path_full = os.path.join(env_settings().network_path, self.net_path)

        # self.net, _ = load_network(net_path_full, backbone_pretrained=False)
        self.net = backbone_resnet.resnet50(output_layers=self.output_layers, pretrained=True)

        if self.use_gpu:
            self.net.cuda()
        self.net.eval()

        # self.iou_predictor = self.net.bb_regressor

        self.layer_stride = {'conv1': 2, 'layer1': 4, 'layer2': 8, 'layer3': 16, 'layer4': 32, 'classification': 16, 'fc': None}
        self.layer_dim = {'conv1': 64, 'layer1': 256, 'layer2': 512, 'layer3': 1024, 'layer4': 2048, 'classification': 256,'fc': None}

        # self.iounet_feature_layers = self.net.bb_regressor_layer

        if isinstance(self.pool_stride, int) and self.pool_stride == 1:
            self.pool_stride = [1]*len(self.output_layers)

        # self.feature_layers = sorted(list(set(self.output_layers + self.iounet_feature_layers)))

        self.mean = torch.Tensor([0.485, 0.456, 0.406]).view(1,-1,1,1)
        self.std = torch.Tensor([0.229, 0.224, 0.225]).view(1,-1,1,1)

    def free_memory(self):
        if hasattr(self, 'net'):
            del self.net
        if hasattr(self, 'iou_predictor'):
            del self.iou_predictor
        if hasattr(self, 'iounet_backbone_features'):
            del self.iounet_backbone_features
        if hasattr(self, 'iounet_features'):
            del self.iounet_features

    def dim(self):
        return TensorList([self.layer_dim[l] for l in self.output_layers])

    def stride(self):
        return TensorList([s * self.layer_stride[l] for l, s in zip(self.output_layers, self.pool_stride)])

    def extract(self, im: torch.Tensor):
        im = im/255
        im -= self.mean
        im /= self.std

        if self.use_gpu:
            im = im.cuda()

        with torch.no_grad():
            #output_features = self.net.extract_features(im, self.feature_layers)
            output_features = self.net(im, output_layers=self.output_layers)

        # # Store the raw resnet features which are input to iounet
        # self.iounet_backbone_features = TensorList([output_features[layer].clone() for layer in self.iounet_feature_layers])
        #
        # # Store the processed features from iounet, just before pooling
        # with torch.no_grad():
        #     self.iounet_features = TensorList(self.iou_predictor.get_iou_feat(self.iounet_backbone_features))

        return TensorList([output_features[layer] for layer in self.output_layers])

