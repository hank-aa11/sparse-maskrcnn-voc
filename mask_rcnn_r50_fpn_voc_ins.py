_base_ = '/mnt/data/jichuan/openmmlab_voc_project/mmdetection/configs/mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py'

dataset_type = 'CocoDataset'
data_root = '/mnt/data/jichuan/openmmlab_voc_project/data/voc_ins/'
backend_args = None 

metainfo = dict(
    classes=(
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
        'bus', 'car', 'cat', 'chair', 'cow',
        'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'),
    palette=[
        (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
        (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
        (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
        (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)
    ]
)

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True, backend_args=backend_args), 
    dict(type='RandomResize', scale=[(1333, 640), (1333, 800)], keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion',
         brightness_delta=32,
         contrast_range=(0.5, 1.5),
         saturation_range=(0.5, 1.5),
         hue_delta=18),
    dict(type='Pad', size_divisor=32),
    dict(type='PackDetInputs')
]

val_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True, backend_args=backend_args),
    dict(type='Pad', size_divisor=32),
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')) 
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='Pad', size_divisor=32),
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]

tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale=(1333, 800), keep_ratio=True),
                dict(type='Resize', scale=(1333, 640), keep_ratio=True)
            ],
            [
                dict(type='RandomFlip', prob=0.0, direction='horizontal'),
                dict(type='RandomFlip', prob=1.0, direction='horizontal')
            ],
            [dict(type='Pad', size_divisor=32)],
            [dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'flip', 'flip_direction'))] # TTA时可能需要更多meta_keys
        ])
]

train_dataloader = dict(
    batch_size=2, 
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_train0712.json', 
        data_prefix=dict(img='train0712/'), 
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        metainfo=metainfo,
        pipeline=train_pipeline)
)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_val07.json',
        data_prefix=dict(img='val07/'),
        test_mode=True,
        metainfo=metainfo,
        pipeline=val_pipeline)
)

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_val07.json',
        data_prefix=dict(img='val07/'),
        test_mode=True,
        metainfo=metainfo,
        pipeline=tta_pipeline)
)

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_val07.json',
    metric=['bbox', 'segm'],
    backend_args=backend_args) 
test_evaluator = val_evaluator

model = dict(
    backbone=dict(
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
    ),
    neck=dict(
        norm_cfg=dict(type='GN', num_groups=32)
    ),
    roi_head=dict(
        bbox_roi_extractor=dict(roi_layer=dict(output_size=7, sampling_ratio=2)),
        mask_roi_extractor=dict(roi_layer=dict(output_size=14, sampling_ratio=2)),
        bbox_head=dict(
            norm_cfg=dict(type='GN', num_groups=32),
            num_classes=20,
            reg_decoded_bbox=True,
            loss_bbox=dict(type='GIoULoss', loss_weight=10.0) 
        ),
        mask_head=dict(
            num_classes=20,
            upsample_cfg=dict(type='bilinear', scale_factor=2),
            loss_mask=dict(type='CrossEntropyLoss', use_mask=True, loss_weight=1.0) 
        ),
        train_cfg=dict( 
            rcnn=dict(
                sampler=dict(
                    type='OHEMSampler', 
                    num=512,
                    pos_fraction=0.5, 
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True
                )
            )
        )
    )
)

max_epochs = 150

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=2,
    dynamic_intervals=[(max_epochs - 6, 1)] 
)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop') 

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=1e-4,  
        betas=(0.9, 0.999),
        weight_decay=0.05 
    ),
    paramwise_cfg=dict(
        norm_decay_mult=0.,
        custom_keys={ 
            'backbone': dict(lr_mult=0.1) 
        }
    )
)

param_scheduler = [
    dict(
        type='LinearLR', 
        start_factor=0.001, 
        by_epoch=False,    
        begin=0, 
        end=1000            
    ),
    dict(
        type='CosineAnnealingLR',
        T_max=max_epochs,     
        eta_min=1e-5,        
        by_epoch=True,        
        begin=0,            
        end=max_epochs
    )
]

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50), 
    param_scheduler=dict(type='ParamSchedulerHook'), 
    checkpoint=dict(type='CheckpointHook', interval=2, max_keep_ckpts=3, save_best='auto', rule='greater'), 
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook', draw=False, interval=100) 
)

env_cfg = dict(
    cudnn_benchmark=False, 
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend', save_dir='./tb_logs/voc_mask_rcnn_tuned_v2')
    ],
    name='visualizer'
)

load_from = '/mnt/data/jichuan/openmmlab_voc_project/checkpoints/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth'(若要从头训练，这句需要注释掉）
log_level = 'INFO'
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
