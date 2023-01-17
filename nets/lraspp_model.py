from collections import OrderedDict

from typing import Dict

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from .mobilenet_backbone import mobilenet_v3_large


class IntermediateLayerGetter(nn.ModuleDict):
    """
    Module wrapper that returns intermediate layers from a model

    It has a strong assumption that the modules have been registered
    into the model in the same order as they are used.
    This means that one should **not** reuse the same nn.Module
    twice in the forward if you want this to work.

    Additionally, it is only able to query submodules that are directly
    assigned to the model. So if `model` is passed, `model.feature1` can
    be returned, but not `model.feature1.layer2`.

    Args:
        model (nn.Module): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
    """

    _version = 2
    __annotations__ = {
        "return_layers": Dict[str, str],
    }

    def __init__(self, model: nn.Module, return_layers: Dict[str, str]) -> None:
        if not set(return_layers).issubset(
            [name for name, _ in model.named_children()]
        ):
            raise ValueError("return_layers are not present in model")
        orig_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()}

        # 重新构建backbone，将没有使用到的模块全部删掉
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break
        # layers={OrderedDict:17} {'0':ConvBNActivation,'1':InvertedResidual, '2':InvertedResidual, ... , '15':InvertedResidual, '16':ConvBNActivation}
        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        out = OrderedDict()  # self = {IntermediateLayerGetter:17}
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out  # out = {OrderedDoct:2} {'low':Tensor(bs, 40, 60, 60), 'high':Tensor(bs, 960, 30, 30)}


class LRASPP(nn.Module):
    """
    Implements a Lite R-ASPP Network for semantic segmentation from
    `"Searching for MobileNetV3"
    <https://arxiv.org/abs/1905.02244>`_.

    Args:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "high" for the high level feature map and "low" for the low level feature map.
        low_channels (int): the number of channels of the low level features.
        high_channels (int): the number of channels of the high level features.
        num_classes (int): number of output classes of the model (including the background).
        inter_channels (int, optional): the number of channels for intermediate computations.
    """

    __constants__ = ["aux_classifier"]

    def __init__(
        self,
        backbone: nn.Module,
        low_channels: int,
        high_channels: int,
        num_classes: int,
        inter_channels: int = 128,
    ) -> None:
        super(LRASPP, self).__init__()
        self.backbone = backbone
        self.classifier = LRASPPHead(
            low_channels, high_channels, num_classes, inter_channels
        )

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        input_shape = x.shape[-2:]
        features = self.backbone(
            x
        )  # feature={OrderedDict:2} {'low':Tensor(bs, 40, 60, 60), 'high':Tensor(bs, 960, 30, 30)}

        out = self.classifier(
            features
        )  # out:Tensor(bs, 21, 60, 60) high分支对高层feature maps信息进行处理，low分支对底层feature maps信息进行处理，再讲high和low分支的信息进行相加融合

        out = F.interpolate(
            out, size=input_shape, mode="bilinear", align_corners=False
        )  # out:Tensor(bs, 21, 480, 480) 将out的(height, width)上采样回输入图片的大小(480, 480)

        # result = OrderedDict()
        # result["out"] = out
        # return result  # result={OrderedDict:1} {'out':Tensor(bs, 21, 480, 480)}
        return out


class LRASPPHead(nn.Module):
    def __init__(
        self,
        low_channels: int,
        high_channels: int,
        num_classes: int,
        inter_channels: int,
    ) -> None:
        super(LRASPPHead, self).__init__()
        self.cbr = nn.Sequential(
            nn.Conv2d(
                high_channels, inter_channels, kernel_size=1, bias=False
            ),  # 960in, 128out, k1x1
            nn.BatchNorm2d(inter_channels),  # 128
            nn.ReLU(inplace=True),
        )
        self.scale = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Conv2d(high_channels, inter_channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )
        self.low_classifier = nn.Conv2d(
            low_channels, num_classes, kernel_size=1
        )  # 调整输出的channels到num_classes
        self.high_classifier = nn.Conv2d(
            inter_channels, num_classes, kernel_size=1
        )  # 调整输出的channels到num_classes

    def forward(self, inputs: Dict[str, Tensor]) -> Tensor:
        low = inputs["low"]  # low(bs, 40, 60, 60)
        high = inputs["high"]  # high(bs, 960, 30, 30)

        x = self.cbr(high)  # 融合feature extractor high主分支的特征 调整通道数
        s = self.scale(high)  # SE通道注意力机制
        x = x * s  # SE通路与high主分支相乘
        x = F.interpolate(
            x, size=low.shape[-2:], mode="bilinear", align_corners=False
        )  # 将新high主分支进行双线性插值上采样（将(heigh, width)采样到(60,60)）

        return self.low_classifier(low) + self.high_classifier(x)  # 融合high和low分支的信息


def lraspp_mobilenetv3_large(num_classes=21, pretrain_backbone=False, backbone_path=""):
    # 'mobilenetv3_large_imagenet': 'https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth'
    # 'lraspp_mobilenet_v3_large_coco': 'https://download.pytorch.org/models/lraspp_mobilenet_v3_large-d234d4ea.pth'
    backbone = mobilenet_v3_large(
        dilated=True
    )  # 在backbone的最后几层网络中的depthwise逐通道卷积使用膨胀卷积，膨胀卷积系数为r2

    if pretrain_backbone and backbone_path != "":
        # 载入mobilenetv3 large backbone预训练权重
        backbone.load_state_dict(torch.load(backbone_path, map_location="cpu"))

    backbone = backbone.features  # feature extractor

    # Gather the indices of blocks which are strided. These are the locations of C1, ..., Cn-1 blocks.
    # The first and last blocks are always included because they are the C0 (conv1) and Cn.  # getattr() 函数用于返回一个对象属性值
    stage_indices = (
        [0]
        + [i for i, b in enumerate(backbone) if getattr(b, "is_strided", False)]
        + [len(backbone) - 1]
    )
    low_pos = stage_indices[-4]  # use C2 here which has output_stride = 8
    high_pos = stage_indices[-1]  # use C5 which has output_stride = 16
    low_channels = backbone[low_pos].out_channels  # 40
    high_channels = backbone[high_pos].out_channels  # 960

    # 重新构造backbone
    return_layers = {str(low_pos): "low", str(high_pos): "high"}
    backbone = IntermediateLayerGetter(model=backbone, return_layers=return_layers)
    # backbone={Sequential:17}, return_layers={dict:2}{'4':'low', '16':'high'}
    # {IntermediateLayerGetter:17}, 40, 960, 21

    model = LRASPP(backbone, low_channels, high_channels, num_classes)
    return model
