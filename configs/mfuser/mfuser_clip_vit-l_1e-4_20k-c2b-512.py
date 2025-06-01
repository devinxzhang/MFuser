_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_20k.py',
    '../_base_/datasets/city2bdd-512.py']

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=8)

model = dict(
    type='MFuser_CLIP',
    pretrained='pretrained/ViT-L-14-336px.pt',
    token_embed_dim=768,
    text_dim=768,
    context_length=5,
    backbone=dict(
        type='CLIPVisionTransformer',
        patch_size=16,
        width=1024,
        output_dim=768,
        get_embeddings=True,
        drop_path_rate=0.1,
        layers=24,
        input_resolution=512,
        style='pytorch',
        out_indices=[7, 11, 15, 23],
        heads=16,
        ignore_last_attn=False),
    text_encoder=dict(
        type='CLIPTextContextEncoder',
        context_length=10,
        embed_dim=768,
        transformer_width=768,
        transformer_heads=12,
        transformer_layers=12,
        style='pytorch'),
    context_decoder=dict(
        type='ContextDecoder',
        transformer_width=256,
        transformer_heads=4,
        transformer_layers=1,
        visual_dim=768,
        dropout=0.1,
        outdim=768,
        style='pytorch'),
    decode_head=dict(
        type='tqdmHead',
        in_channels=[2048, 2048, 2048, 2048],
        feat_channels=256,
        out_channels=256,
        in_index=[0, 1, 2, 3],
        num_things_classes=8,
        num_stuff_classes=11,
        num_queries=19,
        num_transformer_feat_level=3,
        pixel_decoder=dict(
            type='tqdmMSDeformAttnPixelDecoder',
            num_text_embeds=19,
            num_outs=3,
            norm_cfg=dict(type='GN', num_groups=32),
            act_cfg=dict(type='ReLU'),
            encoder=dict(
                type='DetrTransformerDecoder',
                return_intermediate=True,
                num_layers=6,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict( # for self attention
                            type='MultiScaleDeformableAttention',
                            embed_dims=256,
                            num_heads=8,
                            num_levels=3,
                            num_points=4,
                            im2col_step=64,
                            dropout=0.0,
                            batch_first=False,
                            norm_cfg=None, 
                            init_cfg=None
                        ),
                        dict( # for cross attention
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            attn_drop=0.0,
                            proj_drop=0.0,
                            dropout_layer=None,
                            batch_first=False
                        )
                    ],
                    ffn_cfgs=dict(
                        embed_dims=256,
                        feedforward_channels=1024, # reduced from 2048,
                        num_fcs=2,
                        act_cfg=dict(type='ReLU', inplace=True),
                        ffn_drop=0.0,
                        dropout_layer=None,
                        add_identity=True),
                    feedforward_channels=2048,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                    'ffn', 'norm')), # same order with mask2former / 
                init_cfg=None),
            positional_encoding=dict(
                type='SinePositionalEncoding', num_feats=128, normalize=True),
            init_cfg=None),
        enforce_decoder_input_project=False,
        positional_encoding=dict(
            type='SinePositionalEncoding', num_feats=128, normalize=True),
        transformer_decoder=dict(
            type='DetrTransformerDecoder',
            return_intermediate=True,
            num_layers=9,
            transformerlayers=dict(
                type='DetrTransformerDecoderLayer',
                attn_cfgs=dict(
                    type='MultiheadAttention',
                    embed_dims=256,
                    num_heads=8,
                    attn_drop=0.0,
                    proj_drop=0.0,
                    dropout_layer=None,
                    batch_first=False),
                ffn_cfgs=dict(
                    embed_dims=256,
                    feedforward_channels=2048,
                    num_fcs=2,
                    act_cfg=dict(type='ReLU', inplace=True),
                    ffn_drop=0.0,
                    dropout_layer=None,
                    add_identity=True),
                feedforward_channels=2048,
                operation_order=('cross_attn', 'norm', 'self_attn', 'norm',
                                 'ffn', 'norm')),
            init_cfg=None),
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=2.0,
            reduction='mean',
            class_weight=[1.0] * 19 + [0.1]),
        loss_mask=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=5.0),
        loss_dice=dict(
            type='DiceLoss',
            use_sigmoid=True,
            activate=True,
            reduction='mean',
            naive_dice=True,
            eps=1.0,
            loss_weight=5.0),
        train_cfg=dict(
            num_points=12544,
            oversample_ratio=3.0,
            importance_sample_ratio=0.75,
            assigner=dict(
                type='IdentityAssigner',
                num_cls=19,),
            sampler=dict(type='MaskPseudoSampler')),
        test_cfg=dict(
            panoptic_on=True,
            # For now, the dataset does not support
            # evaluating semantic segmentation metric.
            semantic_on=False,
            instance_on=True,
            # max_per_image is for instance segmentation.
            max_per_image=100,
            iou_thr=0.8,
            # In Mask2Former's panoptic postprocessing,
            # it will filter mask area where score is less than 0.5 .
            filter_low_score=True), 
        text_proj=dict(
            text_in_dim=768,
            text_out_dim=256),
    ),
    identity_head=dict(
        type='IdentityHead',
        in_channels=1,
        channels=1,
        num_classes=1,
        dropout_ratio=0.1,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.)),
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(341, 341))
)

optimizer = dict(type='AdamW', lr=1e-4, weight_decay=1e-4,
                 paramwise_cfg=dict(
                    custom_keys={
                        'text_encoder': dict(lr_mult=0.0),
                        'norm': dict(decay_mult=0.)}))

work_dir = './work_dirs_d'
