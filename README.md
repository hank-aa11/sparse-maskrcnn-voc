# Sparse-MaskRCNN-VOC

This repository contains code and configurations to train, evaluate, and visualize **Mask R-CNN** and **Sparse R-CNN** models on the PASCAL VOC dataset using the MMDetection framework.

## Project Overview

This project demonstrates the configuration, training, testing, and comparative analysis of two state-of-the-art object detection models:

* **Mask R-CNN**: Two-stage detector with instance segmentation branch.
* **Sparse R-CNN**: End-to-end sparse proposal detector.

Both models are trained and evaluated on the PASCAL VOC 2007/2012 datasets using MMDetection v3.3.0.

## Environment Setup

1. **Clone the repository**

```bash
   git clone https://github.com/hank-aa11/sparse-maskrcnn-voc.git
   cd sparse-maskrcnn-voc
   ```
2. **Install dependencies**

```bash
  chmod +x setup.sh
  ./setup.sh

   ```
3. **Directory structure**

   * Place PASCAL VOC under `data/VOCdevkit/`.
   * Convert VOC instances to COCO format and place under `data/voc_ins/` for Mask R-CNN.(use voc2coco_instance.py)

## Data Preparation

* **Mask R-CNN**: Uses COCO annotations (`data/voc_ins/annotations/`).
* **Sparse R-CNN**: Uses standard VOC splits defined in `configs/voc07.py`.

## Training

### Mask R-CNN

```bash
python train_mask_rcnn_voc.py
```

* **Epochs**: 150
* **Batch size**: 2

### Sparse R-CNN

```bash
python ./mmdetection/tools/train.py ./sparse-rcnn_r50_fpn_1x_voc.py --work-dir ./work_dirs/sparse-rcnn_r50_fpn_1x_voc
```

* **Epochs**: 60
* **Batch size**: 2

## Evaluation

### Quantitative Metrics

* **Mask R-CNN** on VOC2007 Test:

  * Detection \$AP\_{50}\$: 89.5%
  * Segmentation \$AP\_{50}\$: 72.9%
* **Sparse R-CNN** on VOC2007 Test:

  * Detection \$AP\_{50}\$: 73.8%

### Visualization

* **Mask R-CNN**: `visualize_voc_results.py` generates proposal vs. final predictions and masks.
 ```bash
      python visualize_voc_results.py --mode test \
        --cfg ./work_dirs/mask_rcnn_r50_fpn_voc_ins/merged_cfg.py \
        --ckpt ./work_dirs/mask_rcnn_r50_fpn_voc_ins/best_coco_bbox_mAP_epoch_147.pth\
        --score_thr 0.3

    python visualize_voc_results.py --mode external \
        --cfg ./work_dirs/mask_rcnn_r50_fpn_voc_ins/merged_cfg.py \
        --ckpt ./work_dirs/mask_rcnn_r50_fpn_voc_ins/best_coco_bbox_mAP_epoch_147.pth \
        --score_thr 0.3
```
* **Sparse R-CNN**: `eval_vis_sparse_rcnn_voc.py` overlays detection boxes on test images.
```bash
    python eval_vis_sparse_rcnn.py
```

## Results Summary

| Model        | Detection AP<sub>50</sub> | Segmentation AP<sub>50</sub> | Epochs | Backbone        |
| ------------ | ------------------------- | ---------------------------- | ------ | --------------- |
| Mask R-CNN   | 89.5%                     | 72.9%                        | 150    | ResNet-50 + FPN |
| Sparse R-CNN | 73.8%                     | N/A                          | 60     | ResNet-50 + FPN |

## Model Weights

* **Mask R-CNN**: [Google Drive Link ➔](https://drive.google.com/file/d/1o3IGgOURPI9j-7_-WPuqV5zomT46c2K5/view?usp=drive_link)
* **Sparse R-CNN**: [Google Drive Link ➔](https://drive.google.com/file/d/1d_IeFGk1W-22yvulg5tTFLsg5vlkQpaM/view?usp=drive_link)

