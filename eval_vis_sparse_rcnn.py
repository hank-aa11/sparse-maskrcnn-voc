from mmdet.utils import register_all_modules
register_all_modules(init_default_scope=True)

from pathlib import Path
import json
import tempfile
import torch
import mmengine
from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.runner.checkpoint import load_checkpoint
from mmdet.registry import MODELS
from mmdet.apis import inference_detector                     
from mmdet.registry import VISUALIZERS 

CFG_PATH = (
    "/mnt/data/jichuan/openmmlab_voc_project/"
    "sparse-rcnn_r50_fpn_1x_voc.py"
)
CKPT_PATH = (
    "/mnt/data/jichuan/openmmlab_voc_project/"
    "work_dirs/sparse-rcnn_r50_fpn_1x_voc/"
    "best_pascal_voc_mAP_epoch_21.pth"
)

OUT_DIR = Path(
    "/mnt/data/jichuan/openmmlab_voc_project/"
    "sparse_rcnn_eval_vis"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

VOC_SAMPLES = [
    "/mnt/data/jichuan/openmmlab_voc_project/data/VOCdevkit/VOC2007/JPEGImages/000170.jpg",
    "/mnt/data/jichuan/openmmlab_voc_project/data/VOCdevkit/VOC2007/JPEGImages/000676.jpg",
    "/mnt/data/jichuan/openmmlab_voc_project/data/VOCdevkit/VOC2007/JPEGImages/000799.jpg",
    "/mnt/data/jichuan/openmmlab_voc_project/data/VOCdevkit/VOC2007/JPEGImages/008222.jpg",
]

EXTRA_IMAGES = [
    "/mnt/data/jichuan/openmmlab_voc_project/extra_imgs/cat.jpg",
    "/mnt/data/jichuan/openmmlab_voc_project/extra_imgs/dog.jpg",
    "/mnt/data/jichuan/openmmlab_voc_project/extra_imgs/person.jpg",
]

SCORE_THR = 0.26

VOC_CLASSES = (
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor',
)


def _patched_ckpt_path(ckpt_path: str) -> str:
    ckpt = torch.load(ckpt_path, map_location='cpu')
    meta = ckpt.setdefault('meta', {})
    ds_meta = meta.get('dataset_meta')

    if isinstance(ds_meta, list):
        meta['dataset_meta'] = {'classes': tuple(ds_meta)}
    elif ds_meta is None:
        meta['dataset_meta'] = {'classes': VOC_CLASSES}
    else: 
        return ckpt_path

    patched = Path(tempfile.gettempdir()) / (Path(ckpt_path).stem + "_patched.pth")
    torch.save(ckpt, patched)
    mmengine.print_log(f"[Info] Checkpoint patched → {patched}", "current")
    return str(patched)


def build_model(cfg_path: str, ckpt_path: str, device: str = "cuda:0"):
    cfg = Config.fromfile(cfg_path)
    model = MODELS.build(cfg.model)
    model.to(device)

    load_checkpoint(model, _patched_ckpt_path(ckpt_path),
                    map_location=device, strict=False)
    
    if not hasattr(model, 'dataset_meta') or model.dataset_meta is None:
        model.dataset_meta = {'classes': VOC_CLASSES}
        
    model.cfg = cfg
    model.eval()
    return model, cfg


def evaluate(cfg_path: str, ckpt_path: str, out_dir: Path):
    cfg = Config.fromfile(cfg_path)
    cfg.load_from = _patched_ckpt_path(ckpt_path)
    cfg.work_dir = str(out_dir / "tmp_runner")
    (out_dir / "tmp_runner").mkdir(parents=True, exist_ok=True)

    runner = Runner.from_cfg(cfg)
    metrics = runner.test()
    with open(out_dir / "voc_test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    mmengine.print_log("[✓] 评测完成 → voc_test_metrics.json", "current")


@torch.no_grad()
def visualize(
    cfg_path: str,
    ckpt_path: str,
    img_list: list,
    out_dir: Path,
    score_thr: float = 0.3,
):
    import mmcv
    model, _ = build_model(cfg_path, ckpt_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    visualizer = VISUALIZERS.build(
        dict(type='DetLocalVisualizer', name='vis', save_dir=None)
    )
    visualizer.dataset_meta = model.dataset_meta

    for p in img_list:
        p = Path(p)
        out_file = out_dir / f"{p.stem}_sparsercnn.jpg"

        img_arr = mmcv.imread(str(p))
        result = inference_detector(model, img_arr)

        visualizer.add_datasample(
            name=p.stem,
            image=img_arr,
            data_sample=result,
            pred_score_thr=score_thr,
            draw_gt=False,
            show=False,
            out_file=str(out_file)
        )
        mmengine.print_log(f"[✓] 可视化完成 → {out_file}", "current")

if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mmengine.print_log(">>> Sparse-R-CNN 评测与可视化开始 …", "current")

    evaluate(CFG_PATH, CKPT_PATH, OUT_DIR)                      
    visualize(CFG_PATH, CKPT_PATH, VOC_SAMPLES,
              #OUT_DIR / "vis_voc", SCORE_THR)                    
    visualize(CFG_PATH, CKPT_PATH, EXTRA_IMAGES,
              OUT_DIR / "vis_extra", SCORE_THR)                 

    mmengine.print_log(
        f"\n全部任务完成！结果目录: {OUT_DIR}\n"
        "├── voc_test_metrics.json\n"
        "├── vis_voc/*.jpg   (4 张 VOC 示例)\n"
        "└── vis_extra/*.jpg (3 张外部示例)\n",
        "current",
    )
