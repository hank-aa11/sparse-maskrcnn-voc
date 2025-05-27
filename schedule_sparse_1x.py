train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=60, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=1000), 
    dict(
        type='MultiStepLR',
        begin=0,
        end=60,  # 总 epochs
        by_epoch=True,
        milestones=[45,55],
        gamma=0.1)
]


optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=2e-5, weight_decay=0.05),  
    clip_grad=dict(max_norm=1, norm_type=2)
)

auto_scale_lr = dict(enable=True, base_batch_size=16)
