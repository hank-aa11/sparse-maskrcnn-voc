# 设置MMDetection VOC实验所需的环境
# 1. 创建并激活Conda环境
conda create -n openmmlab-voc python=3.9 -y
source $(conda info --base)/etc/profile.d/conda.sh
conda activate openmmlab-voc

# 2. 安装PyTorch (cu121, torch2.3.0)
pip3 install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
pip install tensorboard future

# 3. 安装MMEngine和MMCV
pip install -U openmim
mim install mmengine
pip install "mmcv<2.2.0,>=2.0.0rc4"

# 4. 克隆并安装MMDetection
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -v -e.‘
cd..

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

