dataset_type = 'VOCDataset'
data_root = '/mnt/data/jichuan/openmmlab_voc_project/data/VOCdevkit'
backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(800, 600), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='RandomAffine', max_rotate_degree=15.0,
         max_translate_ratio=0.0,
         scaling_ratio_range=(1.0, 1.0),
         max_shear_degree=0.0),  
    dict(type='CutOut', n_holes=3, cutout_shape=(50, 50)),  
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(800, 600), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))  
]

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type='ConcatDataset',
        datasets=[
            dict(
                type='VOCDataset',
                data_root=data_root,
                ann_file='VOC2007/ImageSets/Main/trainval.txt',
                data_prefix=dict(sub_data_root='VOC2007/'),
                filter_cfg=dict(filter_empty_gt=True, min_size=32, bbox_min_size=32),
                pipeline=train_pipeline,
                backend_args=backend_args
            ),
            dict(
                type='VOCDataset',
                data_root=data_root,
                ann_file='VOC2012/ImageSets/Main/trainval.txt',  
                data_prefix=dict(sub_data_root='VOC2012/'),
                filter_cfg=dict(filter_empty_gt=True, min_size=32, bbox_min_size=32),
                pipeline=train_pipeline,
                backend_args=backend_args
            )
        ]
    )
)

val_dataloader = dict(
    batch_size=1, 
    num_workers=1,
    persistent_workers=True,
    drop_last=False,  
    sampler=dict(type='DefaultSampler', shuffle=False), 
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='VOC2007/ImageSets/Main/test.txt', 
        data_prefix=dict(sub_data_root='VOC2007/'),
        test_mode=True, 
        pipeline=test_pipeline,  
        backend_args=backend_args
    )
)

test_dataloader = val_dataloader

val_evaluator = dict(type='VOCMetric', metric='mAP', eval_mode='11points')
test_evaluator = val_evaluator  
