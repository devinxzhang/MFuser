IMG_MEAN = [ v*255 for v in [0.48145466, 0.4578275, 0.40821073]]
IMG_VAR = [ v*255 for v in [0.26862954, 0.26130258, 0.27577711]]
img_norm_cfg = dict(mean=IMG_MEAN, std=IMG_VAR, to_rgb=True)

crop_size = (512, 512)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='Resize', ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='ToMask'),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg', 'gt_masks', 'gt_labels'])]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 1024),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])])]

src_dataset_dict = dict(
    type='SynthiaDataset',
    data_root='data/synthia',
    img_dir='RGB',
    ann_dir='GT/LABELS',
    pipeline=train_pipeline)
    
tgt_dataset_dict = dict(
    type='CityscapesDataset',
    data_root='data/cityscapes',
    img_dir='leftImg8bit/val',
    ann_dir='gtFine/val',
    pipeline=test_pipeline)

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    train=dict(
        type='UGDataset', 
        source=src_dataset_dict,
        rare_class_sampling=dict(
            min_pixels=3000, 
            class_temp=100, 
            min_crop_ratio=0.5)),
    val=tgt_dataset_dict,
    test=tgt_dataset_dict)