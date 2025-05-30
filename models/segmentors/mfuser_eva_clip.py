import os
import copy
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from collections import OrderedDict

from mmseg.ops import resize
from mmseg.core import add_prefix
from mmseg.models import builder
from mmseg.models.builder import SEGMENTORS
from mmseg.models.segmentors.base import BaseSegmentor

from ..backbones.eva_clip import get_backbone
from ..backbones.utils import tokenize


@SEGMENTORS.register_module()
class MFuser_EVA_CLIP(BaseSegmentor):

    def __init__(self,
                 eva_clip,
                 decode_head,
                 class_names,
                 context_length,
                 context_decoder=None,
                 token_embed_dim=512, 
                 text_dim=512,
                 neck=None,
                 identity_head=None,
                 visual_reg=True,
                 textual_reg=True,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 **args):

        super(MFuser_EVA_CLIP, self).__init__(init_cfg)

        self.tau = 0.07
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.num_classes = len(class_names)
        self.class_names = class_names
        self.context_length = context_length
        self.visual_reg = visual_reg
        self.textual_reg = textual_reg

        # build components
        self.backbone, self.text_encoder = get_backbone(**eva_clip)
        self.neck = builder.build_neck(neck) if neck is not None else None
        self.context_decoder = builder.build_backbone(context_decoder) if context_decoder is not None else None
        
        # requires_grad False
        for name, param in self.text_encoder.named_parameters():
            param.requires_grad = False
        for name, param in self.backbone.named_parameters():
            if any(key in name for key in ['adapter', 'fpn', 'adapter_proj']):
                param.requires_grad = True
            else:
                param.requires_grad = False
        # build head
        self.decode_head = builder.build_head(decode_head) if decode_head is not None else None
        self.identity_head = builder.build_head(identity_head) if identity_head is not None else None

        # coop
        prompt_num = self.text_encoder.context_length - self.context_length
        self.texts = torch.cat([tokenize(c, context_length=context_length) for c in class_names]).to('cuda')
        self.contexts = nn.Parameter(torch.randn(1, prompt_num, token_embed_dim))
        self.random_text = nn.Parameter(torch.randn(self.num_classes, 768))
        self.gamma = nn.Parameter(torch.ones(text_dim) * 1e-4)

        nn.init.trunc_normal_(self.contexts)
        nn.init.trunc_normal_(self.gamma)

    def fill_all_templates_ensemble(self, x=''):
        res = []
        for template in VILD_PROMPT:
            res.append(template.format(x))
        return res, len(res) // len(VILD_PROMPT)              
        
    def extract_feat(self, img):
        x = self.backbone.extract_feats(img)
        return x

    def after_extract_feat(self, x):
        x_orig = list(x[:-1])
        global_feat, visual_embeddings = x[-1]
        b_size = global_feat.shape[0]

        visual_context = torch.cat([global_feat, visual_embeddings.flatten(-2).permute(0, 2, 1)], dim=1)
        text_embeddings = self.text_encoder(self.texts, context=self.contexts).expand(b_size, -1, -1)  # torch.Size([1, 19, 768])

        if self.context_decoder is not None:
            text_diff = self.context_decoder(text_embeddings, visual_context)
            text_embeddings = text_embeddings + self.gamma * text_diff
        ret_text_emb = text_embeddings

        visual_embeddings = F.normalize(visual_embeddings, dim=1, p=2)
        text_embeddings = F.normalize(text_embeddings, dim=-1, p=2) # [bs, num_class, num_feats]
        score_map = torch.einsum('bchw,bkc->bkhw', visual_embeddings, text_embeddings)

        ret_visual_emb = torch.einsum('bkhw,bdhw->bkd', score_map.softmax(dim=1), visual_embeddings)

        return x_orig, score_map, ret_text_emb, global_feat, visual_embeddings, ret_visual_emb

    def forward_train(self, img, img_metas, gt_semantic_seg, **kwargs):
        x = self.extract_feat(img)
        x_orig, score_map, text_emb, global_feat, visual_embeddings, visual_emb = self.after_extract_feat(x)
        x = list(self.neck(x_orig)) if self.neck is not None else x_orig

        losses = dict()

        # decode head loss
        loss_decode = self.decode_head.forward_train(
            x, text_emb, img_metas, gt_semantic_seg, self.train_cfg, kwargs['gt_labels'], kwargs['gt_masks'])
        losses.update(add_prefix(loss_decode, 'decode'))
        
        # vision-language regularization
        if self.identity_head is not None:
            loss_score_map = self.identity_head.forward_train(
                score_map/self.tau, img_metas, gt_semantic_seg, self.train_cfg)
            losses.update(add_prefix(loss_score_map, 'scr_map'))
        
        return losses

    def encode_decode(self, img, img_metas):
        x = self.extract_feat(img)
        x_orig, score_map, text_emb, global_feat, visual_embeddings, visual_emb = self.after_extract_feat(x)
        x = list(self.neck(x_orig)) if self.neck is not None else x_orig

        out = self.decode_head.forward_test(
            x, text_emb, img_metas, self.test_cfg)
        out = resize(
            input=out,
            size=img.shape[-2:],
            mode='bilinear',
            align_corners=False)
            
        return out

    def slide_inference(self, img, img_meta, rescale):
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """

        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = img.size()
        num_classes = self.num_classes
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, num_classes, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]          
                crop_seg_logit = self.encode_decode(crop_img, img_meta)
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        if torch.onnx.is_in_onnx_export():
            # cast count_mat to constant while exporting to ONNX
            count_mat = torch.from_numpy(
                count_mat.cpu().detach().numpy()).to(device=img.device)
        preds = preds / count_mat
        if rescale:
            preds = resize(
                preds,
                size=img_meta[0]['ori_shape'][:2],
                mode='bilinear',
                align_corners=False)
        return preds

    def whole_inference(self, img, img_meta, rescale):
        """Inference with full image."""

        seg_logit = self.encode_decode(img, img_meta)
        if rescale:
            # support dynamic shape for onnx
            if torch.onnx.is_in_onnx_export():
                size = img.shape[2:]
            else:
                size = img_meta[0]['ori_shape'][:2]
            seg_logit = resize(
                seg_logit,
                size=size,
                mode='bilinear',
                align_corners=False)
        
        if  torch.isnan(seg_logit).any():
            print('########### find NAN #############')

        return seg_logit

    def inference(self, img, img_meta, rescale):
        """Inference with slide/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output segmentation map.
        """

        assert self.test_cfg.mode in ['slide', 'whole']
        ori_shape = img_meta[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in img_meta)
        if self.test_cfg.mode == 'slide':
            seg_logit = self.slide_inference(img, img_meta, rescale)
        else:
            seg_logit = self.whole_inference(img, img_meta, rescale)
        output = F.softmax(seg_logit, dim=1)
        flip = img_meta[0]['flip']
        if flip:
            flip_direction = img_meta[0]['flip_direction']
            assert flip_direction in ['horizontal', 'vertical']
            if flip_direction == 'horizontal':
                output = output.flip(dims=(3, ))
            elif flip_direction == 'vertical':
                output = output.flip(dims=(2, ))

        return output

    def simple_test(self, img, img_meta, rescale=True):
        """Simple test with single image."""
        seg_logit = self.inference(img, img_meta, rescale)
        seg_pred = seg_logit.argmax(dim=1)
        if self.save_seg_logit is True:
            self.seg_logit = seg_logit.cpu().numpy()
        if torch.onnx.is_in_onnx_export():
            # our inference backend only support 4D output
            seg_pred = seg_pred.unsqueeze(0)
            return seg_pred
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred
    
    def aug_test(self, imgs, img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.inference(imgs[0], img_metas[0], rescale)
        for i in range(1, len(imgs)):
            cur_seg_logit = self.inference(imgs[i], img_metas[i], rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(imgs)
        if self.save_seg_logit is True:
            self.seg_logit = seg_logit.cpu().numpy()
        seg_pred = seg_logit.argmax(dim=1)
        seg_pred = seg_pred.cpu().numpy()
        seg_pred = list(seg_pred)
        return seg_pred