from .aspp_head import ASPPHead
from .da_head import DAHead
from .daformer_head import DAFormerHead
from .dlv2_head import DLV2Head
from .fcn_head import FCNHead
from .isa_head import ISAHead
from .psp_head import PSPHead
from .segformer_head import SegFormerHead
from .sep_aspp_head import DepthwiseSeparableASPPHead
from .uper_head import UPerHead
from .setr_up_head import SETRUPHead
from .fpn_head import FPNHead
from .maskclip_head import MaskClipHead
from .identity_head import IdentityHead
from .mask2former_head import Mask2FormerHead
from .maskformer_head import MaskFormerHead
from .tqdm_head import tqdmHead

__all__ = [
    'FCNHead',
    'PSPHead',
    'ASPPHead',
    'UPerHead',
    'DepthwiseSeparableASPPHead',
    'DAHead',
    'DLV2Head',
    'SegFormerHead',
    'DAFormerHead',
    'ISAHead',
    'SETRUPHead',
    'FPNHead',
    'MaskClipHead',
    'IdentityHead',
    'MaskFormerHead',
    'Mask2FormerHead',
    'tqdmHead',
]
