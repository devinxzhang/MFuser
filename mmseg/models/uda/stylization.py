# The ema model and the domain-mixing are based on:
# https://github.com/vikolss/DACS

import math
import os
import random
from copy import deepcopy

import mmcv
import numpy as np
import torch
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('Agg')

from mmseg.core import add_prefix
from mmseg.models import UDA, build_segmentor
from mmseg.models.uda.uda_decorator import UDADecorator, get_module
from mmseg.models.utils.dacs_transforms import (denorm, get_class_masks,
                                                get_mean_std, strong_transform)
from mmseg.models.utils.visualization import subplotimg
from mmseg.utils.utils import downscale_label_ratio
from mmseg.models.uda.sampler import InfiniteSamplerWrapper
from mmseg.models.uda.photo_wct_batch import PhotoWCT, style_transform, FlatFolderDataset
from PIL import Image, ImageFile
import torch.utils.data as data

import torch.nn as nn
import torch.nn.functional as F

from mmcv.runner import load_checkpoint
from torchvision.utils import save_image

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

@UDA.register_module()
class STYLIZATION(UDADecorator):
    def __init__(self, **cfg):
        super(STYLIZATION, self).__init__(**cfg)

        # base parameters
        self.local_iter = 0
        self.total_iter = cfg['total_iter']
        self.batch_size = cfg['batch_size']
        self.crop_size = cfg['crop_size']
        self.optimizer = None

        # stylization parameters
        self.style_img_folder = cfg['style_img_folder']
        self.style_dataset = FlatFolderDataset(self.style_img_folder, style_transform(self.crop_size))
        self.style_iter = iter(data.DataLoader(self.style_dataset, batch_size=self.batch_size, sampler=InfiniteSamplerWrapper(self.style_dataset), num_workers=4))
        self.p_wct = PhotoWCT()
        self.p_wct.load_state_dict(torch.load('pretrained/photo_wct.pth'))
        self.p_wct = self.p_wct.cuda() 

        for p in self.p_wct.parameters():
            p.required_grad = False
        
        # TLDR hyper parameters
        self.original_weight= cfg['original_weight']
        self.style_weight = cfg['style_weight']
        self.fdist_scale_min_ratio = cfg['fdist_scale_min_ratio']
        self.reg_ratio = cfg['reg_ratio']
        self.disent_ratio = cfg['disent_ratio']
        self.reg_lambda = cfg['reg_lambda']
        self.disent_lambda = cfg['disent_lambda']
        self.reg_layers = cfg['reg_layers']
        self.disent_layers = cfg['disent_layers']
        self.threshold = cfg['threshold']
        
        # initialize ImageNet model
        self.imnet_model = build_segmentor(deepcopy(cfg['model']))

    def get_imnet_model(self):
        return get_module(self.imnet_model)
    
    def train_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """
        self.optimizer = optimizer

        optimizer.zero_grad()
        log_vars = self(**data_batch)
        optimizer.step()

        log_vars.pop('loss', None)  # remove the unnecessary 'loss'
        outputs = dict(
            log_vars=log_vars, num_samples=len(data_batch['img_metas']))
        return outputs

    def get_imnet_model(self):
        return get_module(self.imnet_model)

    def get_gram_matrix(self, f):
        B, C, H, W = f.size()
        f = f.view(B, C, H * W)
        G = torch.bmm(f, f.transpose(1,2)).div(H * W)

        return G

    def feat_dist_gram(self, G1, G2, mask=None):
        B, C, C = G1.size()

        gram_dist = G1-G2

        if mask is not None:
            gram_dist = gram_dist * mask

        feat_diff = torch.norm(gram_dist, p=2)
        # feat_diff = torch.pow(torch.norm(G1-G2, p=2), 2)
        # feat_diff = torch.sum(torch.pow(G1-G2, 2))

        return feat_diff / C
        # return feat_diff / ((C * H * W) ** 2)


    def calc_texture_regularization_loss(self, src_grams, img):
        with torch.no_grad():
            self.get_imnet_model().eval()
            feat_imnet = self.get_imnet_model().extract_feat(img)
            feat_imnet = [f.detach() for f in feat_imnet]

        feat_dist = torch.tensor(0.0).cuda()
        for lay in self.reg_layers:
            imnet_feat_gram = self.get_gram_matrix(feat_imnet[lay])

            feat_dist += self.feat_dist_gram(src_grams[lay], imnet_feat_gram) * self.reg_lambda[lay]
            # if self.local_iter % 50 == 0:
            #     print("textreg lay: {}, dist: {:.4f}".format(lay, self.feat_dist_gram(src_grams[lay], imnet_feat_gram)))

        if self.reg_ratio:
            feat_dist = feat_dist * max(0, self.reg_ratio - self.local_iter / self.total_iter)
        feat_loss, feat_log = self._parse_losses({'loss_feat_dist': feat_dist})
        feat_log.pop('loss', None)
        return feat_loss, feat_log


    def calc_style_disentanlge_loss(self, stylized_src_grams, style, gram_masks):
        random_style_feat = self.get_model().extract_feat(style)
        
        feat_dist = torch.tensor(0.0).cuda()
        for lay in self.disent_layers:
            random_style_gram = self.get_gram_matrix(random_style_feat[lay])

            feat_dist += self.feat_dist_gram(random_style_gram, stylized_src_grams[lay], gram_masks[lay]) * self.disent_lambda[lay]
            # if self.local_iter % 50 == 0:
            #     print("disent lay:{}, dist: {:.4f}".format(lay, self.feat_dist_gram(random_style_gram, stylized_src_grams[lay], gram_masks[lay])))

        if self.disent_ratio:
            feat_dist = feat_dist * max(0, self.disent_ratio - self.local_iter / self.total_iter)
        feat_loss, feat_log = self._parse_losses({'loss_feat_dist': feat_dist})
        feat_log.pop('loss', None)
        return feat_loss, feat_log
    
    def get_gram_mask(self, src_feat, stylized_src_feat):
        gram_masks = []
        src_grams = []
        stylized_src_grams = []

        for lay in [0, 1, 2 ,3]:
            B, C, H, W = src_feat[lay].size()

            G1 = self.get_gram_matrix(src_feat[lay])
            G2 = self.get_gram_matrix(stylized_src_feat[lay])

            with torch.no_grad():
                diff = G2-G1

                mask = torch.zeros(B, C, C)
                mask = diff > self.threshold
                
            gram_masks.append(mask)
            src_grams.append(G1)
            stylized_src_grams.append(G2)
            # if self.local_iter % 50 == 0:
            #     print("num features lay:{}, {}/{}".format(lay, mask.sum().item(), B * C * C))

        return src_grams, stylized_src_grams, gram_masks


    def forward_train(self, img, img_metas, gt_semantic_seg):
        """Forward function for training.
        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        log_vars = {}
        dev = img.device

        # original task loss
        clean_losses = self.get_model().forward_train(img, img_metas, gt_semantic_seg, return_feat=True)
        src_feat = clean_losses.pop('features')
        clean_loss, clean_log_vars = self._parse_losses(clean_losses)
        clean_loss = clean_loss * self.original_weight
        log_vars.update(clean_log_vars)
        clean_loss.backward(retain_graph=True)

        # style transfer module
        with torch.no_grad():
            style = next(self.style_iter)
            style = style.cuda()
            stylized_img = self.p_wct(img, style, np.asarray([]), np.asarray([]))
            stylized_img.detach()
        
        stylized_img_metas = img_metas
        stylized_gt_semanitc_seg = gt_semantic_seg

        # stylized task loss
        stylized_clean_losses = self.get_model().forward_train(stylized_img, stylized_img_metas, stylized_gt_semanitc_seg, return_feat=True)
        stylized_src_feat = stylized_clean_losses.pop('features')
        stylized_clean_loss, stylized_clean_log_vars = self._parse_losses(stylized_clean_losses)
        stylized_clean_loss = stylized_clean_loss * self.style_weight
        log_vars.update(add_prefix(stylized_clean_log_vars, 'stylized'))
        stylized_clean_loss.backward(retain_graph=True)

        # get gram-matrices
        src_grams, stylized_src_grams, gram_masks = self.get_gram_mask(src_feat, stylized_src_feat)

        # texture generalization loss
        style_disentangle_loss, feat_log = self.calc_style_disentanlge_loss(stylized_src_grams, style, gram_masks)
        log_vars.update(add_prefix(feat_log, 'disent'))
        style_disentangle_loss.backward()

        # texture regularization loss
        texture_regularization_loss, feat_log = self.calc_texture_regularization_loss(src_grams, img)
        log_vars.update(add_prefix(feat_log, 'textreg'))
        texture_regularization_loss.backward()

        self.local_iter += 1

        return log_vars
