
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES

@LOSSES.register_module()
class L2Loss(nn.Module):
    def __init__(self,
                reduction='mean',
                loss_weight=1.0):
        super(L2Loss, self).__init__()
        self.loss_weight= loss_weight
        self.creterion = nn.MSELoss(reduction=reduction)
    
    def forward(self, input, target, **kwargs):
        l2loss = self.loss_weight * self.creterion(input, target)
        return l2loss
