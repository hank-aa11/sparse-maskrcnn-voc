# dataset settings
dataset_type = 'VOCDataset'
# 建议使用正斜杠 '/' 作为路径分隔符，兼容性更好
data_root = '/mnt/data/jichuan/openmmlab_voc_project/data/VOCdevkit'

backend_args = None

# === 训练数据处理流水线 ===
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(800, 600), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='RandomAffine', max_rotate_degree=15.0,
         max_translate_ratio=0.0,
         scaling_ratio_range=(1.0, 1.0),
         max_shear_degree=0.0),  # 新增随机旋转
    dict(type='CutOut', n_holes=3, cutout_shape=(50, 50)),  # 新增遮挡增强
    dict(type='PackDetInputs')
]

# === 测试/验证数据处理流水线 ===
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(800, 600), keep_ratio=True),
    # 注意：在 Resize 后加载标注是为了避免标注框也被缩放影响
    # 但在 MMDetection 3.x 中，通常建议在 Resize 前加载标注，PackDetInputs 会处理坐标转换
    # 如果遇到问题，可以尝试将 LoadAnnotations 移到 Resize 之前
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))  # 定义需要传递给模型的元信息
]

# === 训练数据加载器 ===
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type='ConcatDataset',  # 使用数据集拼接
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
                ann_file='VOC2012/ImageSets/Main/trainval.txt',  # 添加 VOC2012
                data_prefix=dict(sub_data_root='VOC2012/'),
                filter_cfg=dict(filter_empty_gt=True, min_size=32, bbox_min_size=32),
                pipeline=train_pipeline,
                backend_args=backend_args
            )
        ]
    )
)

# === 验证数据加载器 ===
val_dataloader = dict(
    batch_size=1,  # 验证时通常 batch_size 为 1
    num_workers=1,
    persistent_workers=True,
    drop_last=False,  # 不丢弃最后一个不完整的 batch
    sampler=dict(type='DefaultSampler', shuffle=False),  # 验证时不打乱数据
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='VOC2007/ImageSets/Main/test.txt',  # *** 使用 VOC2007 的 test 文件进行验证 ***
        data_prefix=dict(sub_data_root='VOC2007/'),
        test_mode=True,  # 设置为测试模式
        pipeline=test_pipeline,  # 使用上面定义的测试流水线
        backend_args=backend_args
    )
)

# === 测试数据加载器 ===
# 通常测试数据加载器与验证数据加载器配置相同
test_dataloader = val_dataloader

# === 评估器配置 ===
# Pascal VOC2007 默认使用 11点插值计算 mAP
val_evaluator = dict(type='VOCMetric', metric='mAP', eval_mode='11points')
test_evaluator = val_evaluator  # 测试评估器与验证评估器相同
