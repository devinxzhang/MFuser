import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from collections import OrderedDict
from mmseg.models.backbones import ResNet
from mmseg.models.builder import BACKBONES
from timm.models.layers import drop_path, trunc_normal_

from ..eva_clip.adapter_module import MVFuser
from ..dino_v2 import DinoVisionTransformer
    

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class QuickGELU(nn.Module):

    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)

class ResidualAttentionBlock(nn.Module):
    
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, drop_path=0.):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor, H=None, W=None):
        x = x + self.drop_path(self.attention(self.ln_1(x)))
        x = x + self.drop_path(self.mlp(self.ln_2(x)))
        return x

class Transformer(nn.Module):

    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, drop_path_rate=0.):
        super().__init__()
        self.width = width
        self.layers = layers
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, layers)]  # stochastic depth decay rule
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask, dpr[i]) for i in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)

class Attention(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)


        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, k, v):
        B, N, C = q.shape
        assert k.shape == v.shape
        B, M, C = k.shape
        q = self.q_proj(q).reshape(B, N, self.num_heads, C // self.num_heads)
        k = self.k_proj(k).reshape(B, M, self.num_heads, C // self.num_heads)
        v = self.v_proj(v).reshape(B, M, self.num_heads, C // self.num_heads)

        attn = torch.einsum('bnkc,bmkc->bknm', q, k) * self.scale

        attn = attn.softmax(dim=-1)

        x = torch.einsum('bknm,bmkc->bnkc', attn, v).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MTEnhancer(nn.Module):

    def __init__(
        self,
        d_model,
        nhead,
        dropout=0.1):
        super().__init__()
        self.self_attn = Attention(d_model, nhead, proj_drop=dropout)
        self.cross_attn = MVFuser(d_model, d_state=16)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x, visual):
        q = k = v = self.norm1(x)
        x = x + self.self_attn(q, k, v)
        x_m_x = self.cross_attn(torch.cat([self.norm2(x), visual, self.norm2(x)], dim=1))
        x = x + x_m_x[:, :x.shape[1], :] + x_m_x[:, -x.shape[1]:, :]
        x = x + self.dropout(self.mlp(self.norm3(x)))  
        
        return x

@BACKBONES.register_module()
class CLIPVisionTransformer(nn.Module):

    def __init__(self, 
                 input_resolution=224,
                 patch_size=32, 
                 width=768, 
                 layers=12, 
                 heads=12, 
                 output_dim=512, 
                 drop_path_rate=0.0, 
                 out_indices=[3, 5, 7, 11], 
                 pretrained=None, 
                 get_embeddings=False, 
                 ignore_last_attn=False, 
                 **kwargs):

        super().__init__()

        self.embed_dim = width
        self.output_dim = output_dim
        self.pretrained = pretrained
        self.patch_size = patch_size
        
        if isinstance(input_resolution, int):
            self.input_resolution = (input_resolution, input_resolution)
        elif isinstance(input_resolution, tuple):
            self.input_resolution = input_resolution       

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.positional_embedding = nn.Parameter(scale * torch.randn((self.input_resolution[0] // patch_size) * (self.input_resolution[1] // patch_size) + 1, width))
        self.spatial_size = (self.input_resolution[0] // patch_size, self.input_resolution[1] // patch_size)
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.ln_pre = LayerNorm(width)
        self.get_embeddings = get_embeddings

        self.transformer = Transformer(width, layers, heads, drop_path_rate=drop_path_rate)

        self.out_indices = out_indices
        self.ignore_last_attn = ignore_last_attn

        if get_embeddings:
            self.ln_post = LayerNorm(width)
            self.proj = nn.Parameter(scale * torch.randn(width, output_dim))      

        self.fpn_dim = width + 1024
        self.fpn1 = nn.Sequential(
                nn.ConvTranspose2d(self.fpn_dim, self.fpn_dim, kernel_size=2, stride=2),
                nn.SyncBatchNorm(self.fpn_dim),
                nn.GELU(),
                nn.ConvTranspose2d(self.fpn_dim, self.fpn_dim, kernel_size=2, stride=2))
        self.fpn2 = nn.Sequential(
            nn.ConvTranspose2d(self.fpn_dim, self.fpn_dim, kernel_size=2, stride=2))
        self.fpn3 = nn.Identity()
        self.fpn4 = nn.MaxPool2d(kernel_size=2, stride=2)      
        
        # DINOv2-L
        self.dinov2 = DinoVisionTransformer(patch_size=16,
                        embed_dim=1024,
                        depth=24,
                        num_heads=16,
                        mlp_ratio=4,
                        img_size=512,
                        ffn_layer="mlp",
                        init_values=1e-05,
                        block_chunks=0,
                        qkv_bias=True,
                        proj_bias=True,
                        ffn_bias=True,)
        dinov2_state_dict = torch.load('pretrained/dinov2_vitl14_pretrain.pth')
        all_keys = list(dinov2_state_dict.keys())
        # interpolate position embedding
        if 'pos_embed' in dinov2_state_dict:
            pos_embed_checkpoint = dinov2_state_dict['pos_embed']
            embedding_size = pos_embed_checkpoint.shape[-1]
            num_patches = self.dinov2.patch_embed.num_patches
            num_extra_tokens = self.dinov2.pos_embed.shape[-2] - num_patches
            # height (== width) for the checkpoint position embedding
            orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
            # height (== width) for the new position embedding
            new_size = int(num_patches ** 0.5)
            # class_token and dist_token are kept unchanged
            if orig_size != new_size:
                print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
                extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
                # only the position tokens are interpolated
                pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
                pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
                pos_tokens = torch.nn.functional.interpolate(
                    pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
                pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
                new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
                dinov2_state_dict['pos_embed'] = new_pos_embed

                patch_embed_proj = dinov2_state_dict['patch_embed.proj.weight']
                patch_size = self.dinov2.patch_embed.patch_size
                dinov2_state_dict['patch_embed.proj.weight'] = torch.nn.functional.interpolate(
                    patch_embed_proj.float(), size=patch_size, mode='bicubic', align_corners=False)
        self.dinov2.load_state_dict(dinov2_state_dict, strict=True)
        self.adapter = nn.Sequential(*[MVFuser(self.embed_dim, d_state=16) for i in range(layers)])
        self.adapter_proj1 = nn.Sequential(*[nn.Linear(1024, self.embed_dim) for i in range(layers)])
        self.adapter_proj2 = nn.Sequential(*[nn.Linear(self.embed_dim, 1024) for i in range(layers)])
        self.adapter_proj3 = nn.Linear(1024, self.embed_dim)

    def init_weights(self, pretrained=None):
        pretrained = pretrained or self.pretrained
        print("backbone:", pretrained)
        if isinstance(pretrained, str):
            checkpoint = torch.jit.load(pretrained, map_location='cpu').float().state_dict()

            state_dict = {}

            for k in checkpoint.keys():
                if k.startswith('visual.'):
                    new_k = k.replace('visual.', '')
                    state_dict[new_k] = checkpoint[k]

            if 'positional_embedding' in state_dict.keys():
                if self.positional_embedding.shape != state_dict['positional_embedding'].shape:
                    print(f'Resize the pos_embed shape from {state_dict["positional_embedding"].shape} to {self.positional_embedding.shape}')
                    cls_pos = state_dict["positional_embedding"][0:1, :]
                    orig_size = int(state_dict["positional_embedding"][1:,].shape[0] ** 0.5)
                    spatial_pos = F.interpolate(state_dict["positional_embedding"][1:,].reshape(1, orig_size, orig_size, self.embed_dim).permute(0, 3, 1, 2), size=self.spatial_size, mode='bilinear')
                    spatial_pos = spatial_pos.reshape(self.embed_dim, self.spatial_size[0]*self.spatial_size[1]).permute(1, 0)
                    positional_embedding = torch.cat([cls_pos, spatial_pos], dim=0)
                    state_dict['positional_embedding'] = positional_embedding
                    assert self.positional_embedding.shape == state_dict['positional_embedding'].shape

            if self.conv1.weight.shape != state_dict['conv1.weight'].shape:
                print(f'Resize the patch_embed shape from {state_dict["conv1.weight"].shape} to {self.conv1.weight.shape}')
                state_dict["conv1.weight"] = F.interpolate(state_dict["conv1.weight"], size=self.conv1.weight.shape[-2:], mode='bilinear')
                assert self.conv1.weight.shape == state_dict['conv1.weight'].shape
                
            u, w = self.load_state_dict(state_dict, False)
            print(u, w, 'are misaligned params in vision transformer')

    def prepare_tokens_with_masks(self, x: torch.Tensor):
        x = self.conv1(x)
        B, C, H, W = x.shape
        x = x.reshape(x.shape[0], x.shape[1], -1) 
        x = x.permute(0, 2, 1)
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)

        pos = self.positional_embedding.to(x.dtype)
        cls_pos = pos[0,:] + self.class_embedding.to(x.dtype)
        spatial_pos = F.interpolate(pos[1:,].reshape(1, self.spatial_size[0], self.spatial_size[1], C).permute(0, 3, 1, 2), size=(H, W), mode='bilinear')
        spatial_pos = spatial_pos.reshape(1, C, H*W).permute(0, 2, 1)
        pos = torch.cat([cls_pos.reshape(1, 1, C), spatial_pos], dim=1)

        x = x + pos
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)
        return x

    def forward(self, x: torch.Tensor, use_adapter=True):
        # Various Foundation Models use different normalization, convert inputs correspondingly
        IMG_MEAN = torch.tensor([ v*255 for v in [0.48145466, 0.4578275, 0.40821073]]).view(1, 3, 1, 1).cuda()
        IMG_STD = torch.tensor([ v*255 for v in [0.26862954, 0.26130258, 0.27577711]]).view(1, 3, 1, 1).cuda()
        original_x = x * IMG_STD + IMG_MEAN
        DINOV2_IMG_MEAN = torch.tensor([v * 255 for v in [0.485, 0.456, 0.406]]).view(1, 3, 1, 1).cuda()
        DINOV2_IMG_STD = torch.tensor([v * 255 for v in [0.229, 0.224, 0.225]]).view(1, 3, 1, 1).cuda()
        normalized_x = (original_x - DINOV2_IMG_MEAN) / DINOV2_IMG_STD        
        dinov2_x = self.dinov2.prepare_tokens_with_masks(normalized_x)
        
        x = self.conv1(x)
        B, C, H, W = x.shape
        x = x.reshape(x.shape[0], x.shape[1], -1) 
        x = x.permute(0, 2, 1)
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)

        pos = self.positional_embedding.to(x.dtype)
        cls_pos = pos[0,:] + self.class_embedding.to(x.dtype)
        spatial_pos = F.interpolate(pos[1:,].reshape(1, self.spatial_size[0], self.spatial_size[1], C).permute(0, 3, 1, 2), size=(H, W), mode='bilinear')
        spatial_pos = spatial_pos.reshape(1, C, H*W).permute(0, 2, 1)
        pos = torch.cat([cls_pos.reshape(1, 1, C), spatial_pos], dim=1)

        x = x + pos
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)

        features = []
        for i, blk in enumerate(self.transformer.resblocks):
            if self.ignore_last_attn:
                mask = torch.empty(x.shape[0], x.shape[0])
                mask.fill_(float('-inf'))
                mask.fill_diagonal_(0)
                self.transformer.resblocks[-1].attn_mask = mask
            x = blk(x)
            dinov2_x = self.dinov2.blocks[i](dinov2_x)
            
            if use_adapter:
                dinov2_eva = self.adapter[i](torch.cat((self.adapter_proj1[i](dinov2_x), x.permute(1,0,2)), dim=1))
                dinov2_x = dinov2_x + self.adapter_proj2[i](dinov2_eva[:, :dinov2_x.shape[1], :])
                x = x + dinov2_eva[:, dinov2_x.shape[1]:, :].permute(1,0,2)
            
            if i in self.out_indices:
                xp = torch.cat([dinov2_x[:, 1:, :].permute(0, 2, 1).reshape(B, -1, H, W).contiguous(), x.permute(1, 0, 2)[:, 1:, :].permute(0, 2, 1).reshape(B, -1, H, W).contiguous()], dim=1)
                features.append(xp.contiguous())
        
        ops = [self.fpn1, self.fpn2, self.fpn3, self.fpn4]
        for i in range(len(features)):
            features[i] = ops[i](features[i])

        if self.get_embeddings:
            x = x.permute(1, 0, 2) + self.adapter_proj3(dinov2_x)
            x = self.ln_post(x)
            x = x @ self.proj
            
            global_embedding = x[:, :1]
            visual_embedding = x[:, 1:].reshape(B, H, W, -1).permute(0, 3, 1, 2) # B C H W

            features.append([global_embedding, visual_embedding])

        return tuple(features)

@BACKBONES.register_module()
class CLIPTextEncoder(nn.Module):

    def __init__(self, context_length=77,
                 vocab_size=49408,
                 transformer_width=512,
                 transformer_heads=8,
                 transformer_layers=12,
                 embed_dim=1024,
                 out_dim=256,
                 pretrained=None, **kwargs):
        super().__init__()

        self.pretrained = pretrained

        self.context_length = context_length

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))

    def init_weights(self, pretrained=None):
        pretrained = pretrained or self.pretrained
        if isinstance(pretrained, str):
            checkpoint = torch.jit.load(pretrained, map_location='cpu').float().state_dict()

            state_dict = {}

            for k in checkpoint.keys():
                if k.startswith('transformer.'):
                    state_dict[k] = checkpoint[k]
                
                if k == 'positional_embedding' or k == 'text_projection' or k.startswith('token_embedding') or k.startswith('ln_final'):
                    if k == 'positional_embedding' and checkpoint[k].size(0) > self.context_length:
                        checkpoint[k] = checkpoint[k][:self.context_length]
                        print('positional_embedding is tuncated from 77 to', self.context_length)
                    state_dict[k] = checkpoint[k]
             
            u, w = self.load_state_dict(state_dict, False)
            print(u, w, 'are misaligned params in text encoder')


    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def forward(self, text):
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        # x = self.out_proj(x)
        return x

@BACKBONES.register_module()
class CLIPTextContextEncoder(nn.Module):
    def __init__(self, context_length=22,
                 vocab_size=49408,
                 transformer_width=512,
                 transformer_heads=8,
                 transformer_layers=12,
                 embed_dim=1024,
                 out_dim=256,
                 pretrained=None, **kwargs):
        super().__init__()

        self.pretrained = pretrained

        self.context_length = context_length

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.embed_dim = embed_dim

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))

    def init_weights(self, pretrained=None):
        pretrained = pretrained or self.pretrained
        print("text_encoder:", pretrained)
        if isinstance(pretrained, str):
            checkpoint = torch.jit.load(pretrained, map_location='cpu').float().state_dict()

            state_dict = {}

            for k in checkpoint.keys():
                if k.startswith('transformer.'):
                    state_dict[k] = checkpoint[k]
                
                if k == 'positional_embedding' or k == 'text_projection' or k.startswith('token_embedding') or k.startswith('ln_final'):
                    if k == 'positional_embedding' and checkpoint[k].size(0) > self.context_length:
                        checkpoint[k] = checkpoint[k][:self.context_length]
                        print('positional_embedding is tuncated from 77 to', self.context_length)
                    state_dict[k] = checkpoint[k]
             
            u, w = self.load_state_dict(state_dict, False)
            print(u, w, 'are misaligned params in text encoder')


    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def forward(self, text, context=None):
        if context is not None:
            x_text = self.token_embedding(text)  # n_clas, n_text, C
            K, N1, C = x_text.shape
            if len(context.shape) == 3:
                B, N2, C = context.shape

                eos_indx = text.argmax(dim=-1) + N2
                eos_indx = eos_indx.reshape(1, K).expand(B, K).reshape(-1)

                x_text = x_text.reshape(1, K, N1, C).expand(B, K, N1, C)
                context = context.reshape(B, 1, N2, C).expand(B, K, N2, C)
            
            elif len(context.shape) == 4:
                B, K, N2, C = context.shape

                eos_indx = text.argmax(dim=-1) + N2
                eos_indx = eos_indx.reshape(1, K).expand(B, K).reshape(-1)

                x_text = x_text.reshape(1, K, N1, C).expand(B, K, N1, C)
            x = torch.cat([x_text[:,:,0:1], context, x_text[:, :, 1:]], dim=2).reshape(B*K, N1+N2, C)
            x = x + self.positional_embedding
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = self.ln_final(x)
            x = x[torch.arange(x.shape[0]), eos_indx] @ self.text_projection
            x = x.reshape(B, K, self.embed_dim) # 1 19 512
            return x
        
        else:
            x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]
            x = x + self.positional_embedding
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = self.ln_final(x)
            x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
            # x = self.out_proj(x)
            return x

@BACKBONES.register_module()
class ContextDecoder(nn.Module):
    def __init__(self,
                 transformer_width=256,
                 transformer_heads=4,
                 transformer_layers=6,
                 visual_dim=1024,
                 dropout=0.1,
                 **kwargs):
        super().__init__()

        self.memory_proj = nn.Sequential(
            nn.LayerNorm(visual_dim),
            nn.Linear(visual_dim, transformer_width),
            nn.LayerNorm(transformer_width),
        )

        self.text_proj = nn.Sequential(
            nn.LayerNorm(visual_dim),
            nn.Linear(visual_dim, transformer_width),
        )

        self.decoder = nn.ModuleList([
                    MTEnhancer(transformer_width, transformer_heads, dropout) for _ in range(transformer_layers)
                ])
        
        self.out_proj = nn.Sequential(
            nn.LayerNorm(transformer_width),
            nn.Linear(transformer_width, visual_dim)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    
    def forward(self, text, visual):
        B, N, C = visual.shape
        visual = self.memory_proj(visual)
        x = self.text_proj(text)

        for layer in self.decoder:
            x = layer(x, visual)
        
        return self.out_proj(x)
