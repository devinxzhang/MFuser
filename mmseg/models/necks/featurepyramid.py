# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.cnn import build_norm_layer
# from mmengine.model import BaseModule


from ..builder import NECKS
from ..utils import resize


@NECKS.register_module()
class Feature2Pyramid(nn.Module):
    """Feature Pyramid Network.

    This neck is the implementation of `Feature Pyramid Networks for Object
    Detection <https://arxiv.org/abs/1612.03144>`_.

    Args:
        in_channels (list[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale).
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level.
        add_extra_convs (bool | str): If bool, it decides whether to add conv
            layers on top of the original feature maps. Default to False.
            If True, its actual mode is specified by `extra_convs_on_inputs`.
            If str, it specifies the source feature map of the extra convs.
            Only the following options are allowed

            - 'on_input': Last feat map of neck inputs (i.e. backbone feature).
            - 'on_lateral': Last feature map after lateral convs.
            - 'on_output': The last output feature map after fpn convs.
        extra_convs_on_inputs (bool, deprecated): Whether to apply extra convs
            on the original feature from the backbone. If True,
            it is equivalent to `add_extra_convs='on_input'`. If False, it is
            equivalent to set `add_extra_convs='on_output'`. Default to True.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Default: False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Default: False.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (dict): Config dict for activation layer in ConvModule.
            Default: None.
        upsample_cfg (dict): Config dict for interpolate layer.
            Default: dict(mode='nearest').
        init_cfg (dict or list[dict], optional): Initialization config dict.

    Example:
        >>> import torch
        >>> in_channels = [2, 3, 5, 7]
        >>> scales = [340, 170, 84, 43]
        >>> inputs = [torch.rand(1, c, s, s)
        ...           for c, s in zip(in_channels, scales)]
        >>> self = FPN(in_channels, 11, len(in_channels)).eval()
        >>> outputs = self.forward(inputs)
        >>> for i in range(len(outputs)):
        ...     print(f'outputs[{i}].shape = {outputs[i].shape}')
        outputs[0].shape = torch.Size([1, 11, 340, 340])
        outputs[1].shape = torch.Size([1, 11, 170, 170])
        outputs[2].shape = torch.Size([1, 11, 84, 84])
        outputs[3].shape = torch.Size([1, 11, 43, 43])
    """

    def __init__(self,
                 embed_dim,
                 rescales=[4, 2, 1, 0.5],
                 norm_cfg=dict(type='SyncBN', requires_grad=True) ):
        super(Feature2Pyramid, self).__init__() # init_cfg
        assert isinstance(in_channels, list)
        self.rescales = rescales
        self.upsample_4x = None
        for k in self.rescales:
            if k==4:
                self.upsample_4x = nn.Sequential(
                    nn.ConvTranspose2d(
                        embed_dim, embed_dim, kernerl_size=2, stride=2),
                    build_norm_layer(norm_cfg, embed_dim, kernel_size=2, stride=2),
                    nn.GELU(),
                    nn.ConvTransposed2d(
                        embed_dim, embed_dim, kernel_size=2, stride=2),
                )
            elif k == 2:
                self.upsample_2x = nn.Sequential(
                    nn.ConvTransposed2d(
                        embed_dim, embed_dim, kernel_size=2, stride=2)
                )
            elif k == 1:
                self.identity = nn.Identity()
            elif k == 0.5:
                self.downsample_2x = nn.MaxPool2d(kernel_size=2, stride=2)
            else :
                raise KeyError(f'invalid {k} for feature2pyramid')


    def forward(self, inputs):
        assert len(inputs) == len(self.rescales)

        outputs = []
        if len(rescales)==4:
            ops = [self.upsample_4x, self.upsample_4x, self.identity, self.downsample_2x]
        elif len(rescales)==3:
            ops = [self.upsample_4x, self.upsample_4x, self.identity]
        for i in range(len(inputs)):
            outputs.append(ops[i](inputs[i]))

        return tuple(outs)
