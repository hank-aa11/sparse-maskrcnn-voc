#!/usr/bin/env python
import os
from mmengine.config import Config
from mmengine.runner import Runner

def main():
    cfg = Config.fromfile('./mask_rcnn_r50_fpn_voc_ins.py')
    cfg.load_from = (
        '/mnt/data/jichuan/openmmlab_voc_project/checkpoints/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth'
    )
    cfg.work_dir = './work_dirs/mask_rcnn_r50_fpn_voc_ins'
    os.makedirs(cfg.work_dir, exist_ok=True)
    cfg.dump(os.path.join(cfg.work_dir, 'merged_cfg.py'))

    runner = Runner.from_cfg(cfg)
    runner.train()

if __name__ == '__main__':
    main()
