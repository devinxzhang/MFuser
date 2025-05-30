"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import transforms
from pathlib import Path
from mmseg.models.uda.vgg import VGGEncoder, VGGDecoder

def style_transform(size):
    transform_list = []
    transform_list.append(transforms.Resize((size, size)))
    transform_list.append(transforms.ToTensor())
    return transforms.Compose(transform_list)


class FlatFolderDataset(data.Dataset):
    def __init__(self, root, transform, name=False):
        super(FlatFolderDataset, self).__init__()
        self.root = root
        self.paths = list(Path(self.root).glob('*'))
        self.transform = transform
        self.name = name

    def __getitem__(self, index):
        if self.name is False:
            path = self.paths[index]
            img = Image.open(str(path)).convert('RGB')
            img = self.transform(img)
            return img
        else:
            path = self.paths[index]
            img = Image.open(str(path)).convert('RGB')
            img = self.transform(img)
            img_name = str(path).split('/')[-1]
            return [img, img_name]

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'FlatFolderDataset'

class Timer:
    def __init__(self, msg):
        self.msg = msg
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_value, exc_tb):
        print(self.msg % (time.time() - self.start_time))


class PhotoWCT(nn.Module):
    def __init__(self):
        super(PhotoWCT, self).__init__()
        self.e1 = VGGEncoder(1)
        self.d1 = VGGDecoder(1)
        self.e2 = VGGEncoder(2)
        self.d2 = VGGDecoder(2)
        self.e3 = VGGEncoder(3)
        self.d3 = VGGDecoder(3)
        self.e4 = VGGEncoder(4)
        self.d4 = VGGDecoder(4)
    
    def forward(self, cont_img, styl_img, cont_seg, styl_seg):
        self.__compute_label_info(cont_seg, styl_seg)

        sF4, sF3, sF2, sF1 = self.e4.forward_multiple(styl_img)

        cF4, cpool_idx, cpool1, cpool_idx2, cpool2, cpool_idx3, cpool3 = self.e4(cont_img)
        sF4 = sF4.data.squeeze(0)
        cF4 = cF4.data.squeeze(0)

        csF4 = self.__feature_wct(cF4, sF4, cont_seg, styl_seg)
        # print(csF4.shape)
        Im4 = self.d4(csF4, cpool_idx, cpool1, cpool_idx2, cpool2, cpool_idx3, cpool3)
        
        cF3, cpool_idx, cpool1, cpool_idx2, cpool2 = self.e3(Im4)
        sF3 = sF3.data.squeeze(0)
        cF3 = cF3.data.squeeze(0)
        csF3 = self.__feature_wct(cF3, sF3, cont_seg, styl_seg)
        Im3 = self.d3(csF3, cpool_idx, cpool1, cpool_idx2, cpool2)

        cF2, cpool_idx, cpool = self.e2(Im3)
        sF2 = sF2.data.squeeze(0)
        cF2 = cF2.data.squeeze(0)
        csF2 = self.__feature_wct(cF2, sF2, cont_seg, styl_seg)
        Im2 = self.d2(csF2, cpool_idx, cpool)

        cF1 = self.e1(Im2)
        sF1 = sF1.data.squeeze(0)
        cF1 = cF1.data.squeeze(0)
        csF1 = self.__feature_wct(cF1, sF1, cont_seg, styl_seg)
        Im1 = self.d1(csF1)
        
        return Im1
        
    def __compute_label_info(self, cont_seg, styl_seg):
        if cont_seg.size == False or styl_seg.size == False:
            return
        max_label = np.max(cont_seg) + 1
        self.label_set = np.unique(cont_seg)
        self.label_indicator = np.zeros(max_label)
        for l in self.label_set:
            # if l==0:
            #   continue
            is_valid = lambda a, b: a > 10 and b > 10 and a / b < 100 and b / a < 100
            o_cont_mask = np.where(cont_seg.reshape(cont_seg.shape[0] * cont_seg.shape[1]) == l)
            o_styl_mask = np.where(styl_seg.reshape(styl_seg.shape[0] * styl_seg.shape[1]) == l)
            self.label_indicator[l] = is_valid(o_cont_mask[0].size, o_styl_mask[0].size)

    def __feature_wct(self, cont_feat, styl_feat, cont_seg, styl_seg):
        batch, cont_c, cont_h, cont_w = cont_feat.size(0), cont_feat.size(1), cont_feat.size(2), cont_feat.size(3)
        batch, styl_c, styl_h, styl_w = styl_feat.size(0), styl_feat.size(1), styl_feat.size(2), styl_feat.size(3)
        cont_feat_view = cont_feat.view(batch, cont_c, -1).clone()
        styl_feat_view = styl_feat.view(batch, styl_c, -1).clone()

        if cont_seg.size == False or styl_seg.size == False:
            target_feature = self.__wct_core(cont_feat_view, styl_feat_view)
        else:
            target_feature = cont_feat.view(cont_c, -1).clone()
            if len(cont_seg.shape) == 2:
                t_cont_seg = np.asarray(Image.fromarray(cont_seg).resize((cont_w, cont_h), Image.NEAREST))
            else:
                t_cont_seg = np.asarray(Image.fromarray(cont_seg, mode='RGB').resize((cont_w, cont_h), Image.NEAREST))
            if len(styl_seg.shape) == 2:
                t_styl_seg = np.asarray(Image.fromarray(styl_seg).resize((styl_w, styl_h), Image.NEAREST))
            else:
                t_styl_seg = np.asarray(Image.fromarray(styl_seg, mode='RGB').resize((styl_w, styl_h), Image.NEAREST))

            for l in self.label_set:
                if self.label_indicator[l] == 0:
                    continue
                cont_mask = np.where(t_cont_seg.reshape(t_cont_seg.shape[0] * t_cont_seg.shape[1]) == l)
                styl_mask = np.where(t_styl_seg.reshape(t_styl_seg.shape[0] * t_styl_seg.shape[1]) == l)
                if cont_mask[0].size <= 0 or styl_mask[0].size <= 0:
                    continue

                cont_indi = torch.LongTensor(cont_mask[0])
                styl_indi = torch.LongTensor(styl_mask[0])
                if self.is_cuda:
                    cont_indi = cont_indi.cuda(0)
                    styl_indi = styl_indi.cuda(0)

                cFFG = torch.index_select(cont_feat_view, 1, cont_indi)
                sFFG = torch.index_select(styl_feat_view, 1, styl_indi)
                # print(len(cont_indi))
                # print(len(styl_indi))
                tmp_target_feature = self.__wct_core(cFFG, sFFG)
                # print(tmp_target_feature.size())
                if torch.__version__ >= "0.4.0":
                    # This seems to be a bug in PyTorch 0.4.0 to me.
                    new_target_feature = torch.transpose(target_feature, 1, 0)
                    new_target_feature.index_copy_(0, cont_indi, \
                            torch.transpose(tmp_target_feature,1,0))
                    target_feature = torch.transpose(new_target_feature, 1, 0)
                else:
                    target_feature.index_copy_(1, cont_indi, tmp_target_feature)

        target_feature = target_feature.view_as(cont_feat)
        ccsF = target_feature.float()
        return ccsF
    
    def __wct_core(self, cont_feat, styl_feat):
        cFSize = cont_feat.size()
        #print(cFSize)
        c_mean = torch.mean(cont_feat, 2)  # b x c x (h x w)
        c_mean = c_mean.unsqueeze(2).expand_as(cont_feat)
        cont_feat = cont_feat - c_mean

        # iden = torch.eye(cFSize[1]).to(cont_feat.device)
        contentConv = torch.bmm(cont_feat, cont_feat.transpose(1, 2)).div(cFSize[2] - 1)
        # iden = iden.unsqueeze(0).expand_as(contentConv)
        # contentConv = contentConv + iden
        # print(cont_feat.shape, contentConv.shape)
        try:
            c_u, c_e, v = torch.linalg.svd(contentConv, full_matrices=True)
            c_v = v.mH
        except:
            c_u, c_e, v = torch.linalg.svd(contentConv + 1e-4*contentConv.mean()*torch.rand_like(contentConv), full_matrices=True)
            # c_u, c_e, v = torch.svd(contentConv + 1e-4*contentConv.mean()*torch.rand_like(contentConv))
            c_v = v.mH

        # print(c_u.shape, c_e.shape, c_v.shape)
        # c_e2, c_v = torch.eig(contentConv, True)
        # c_e = c_e2[:,0]
        k_c_list = [cFSize[1]] * cFSize[0]
        # print(k_c_list)
        for j, k_c in enumerate(k_c_list):
            for i in range(cFSize[1] - 1, -1, -1):
                if c_e[j][i] >= 0.00001:
                    k_c = i + 1
                    k_c_list[j] = k_c
                    break
        # print(k_c_list)
        
        sFSize = styl_feat.size()
        s_mean = torch.mean(styl_feat, 2)
        styl_feat = styl_feat - s_mean.unsqueeze(2).expand_as(styl_feat)
        styleConv = torch.bmm(styl_feat, styl_feat.transpose(1, 2)).div(sFSize[2] - 1)
        s_u, s_e, s_v = torch.svd(styleConv, some=False)

        k_s_list = [sFSize[1]] * sFSize[0]
        for j, k_s in enumerate(k_s_list):
            for i in range(sFSize[1] - 1, -1, -1):
                if s_e[j][i] >= 0.00001:
                    k_s = i + 1
                    k_s_list[j] = k_s
                    break
        k_c = min(k_c_list)
        c_d = (c_e[:, 0:k_c]).pow(-0.5)
        step1 = torch.bmm(c_v[:, :, 0:k_c], torch.diag_embed(c_d))
        step2 = torch.bmm(step1, (c_v[:, :, 0:k_c].transpose(1, 2)))
        whiten_cF = torch.bmm(step2, cont_feat)

        s_d = (s_e[:, 0:k_s]).pow(0.5)
        targetFeature = torch.bmm(torch.bmm(torch.bmm(s_v[:, :, 0:k_s], torch.diag_embed(s_d)), (s_v[:, :, 0:k_s].transpose(1, 2))), whiten_cF)
        # print(targetFeature.shape, s_mean.shape)
        targetFeature = targetFeature + s_mean.unsqueeze(2).expand_as(targetFeature)
        return targetFeature
    
    @property
    def is_cuda(self):
        return next(self.parameters()).is_cuda

    # def forward(self, *input):
    #     pass
