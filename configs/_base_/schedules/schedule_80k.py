optimizer_config = dict()
lr_config = dict(policy='poly', power=0.9, min_lr=1e-6, by_epoch=False,
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6)
runner = dict(type='IterBasedRunner', max_iters=40000)
checkpoint_config = dict(by_epoch=False, interval=10000)
evaluation = dict(interval=2000, metric='mIoU')