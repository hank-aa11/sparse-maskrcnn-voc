# 修改mmdetection
  首先需要将./mmdetection/mmdet/models/detectors/two_stage.py中的文件替换成提供的代码，最终可视化的时候才能得到Mask R-CNN第一阶段产生的proposal box和最终的预测结果

# 训练mask-rcnn
    python train-mask-rcnn-voc.py

# 测试mask-rcnn
    python visualize_voc_results.py --mode test \
        --cfg ./work_dirs/mask_rcnn_r50_fpn_voc_ins/merged_cfg.py \
        --ckpt ./work_dirs/mask_rcnn_r50_fpn_voc_ins/best_coco_bbox_mAP_epoch_35.pth\
        --score_thr 0.3

    python visualize_voc_results.py --mode external \
        --cfg ./work_dirs/mask_rcnn_r50_fpn_voc_ins/merged_cfg.py \
        --ckpt ./work_dirs/mask_rcnn_r50_fpn_voc_ins/best_coco_bbox_mAP_epoch_35.pth \
        --score_thr 0.3
