"""
生成 COCO-style JSON（带实例分割掩码）

训练集：VOC2007 train+val  + VOC2012 train+val   → instances_train0712.json
验证集：VOC2007 val                              → instances_val07.json

python voc2coco_instance.py --voc-root data/VOCdevkit --out data/voc_ins
"""
import argparse, os, json, tqdm, xml.etree.ElementTree as ET
from pathlib import Path
import cv2
import numpy as np
from pycocotools import mask as mask_utils

CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

SPLIT_PLAN = {
    '2007': ['train', 'val'],   
    '2012': ['train', 'val'],  
}
VAL_SPLIT = ('2007', 'val')     


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--voc-root', required=True,
                   help='VOCdevkit 根目录，里面应该有 VOC2007/ VOC2012/')
    p.add_argument('--out', required=True,
                   help='输出目录（自动创建）')
    return p.parse_args()


def mask_from_img(mask_img: np.ndarray, instance_id: int):
    m = (mask_img == instance_id).astype(np.uint8)
    if m.sum() == 0:
        return None
    rle = mask_utils.encode(np.asfortranarray(m))
    rle['counts'] = rle['counts'].decode()
    return rle


def iter_split_ids(voc_root: Path, year: str, split: str):
    txt = voc_root / f'VOC{year}' / 'ImageSets' / 'Segmentation' / f'{split}.txt'
    return txt.read_text().strip().split()


def process_image(voc_root: Path, year: str, iid: str,
                  img_id: int, ann_id_start: int):
    jpg = voc_root / f'VOC{year}' / 'JPEGImages' / f'{iid}.jpg'
    png = voc_root / f'VOC{year}' / 'SegmentationObject' / f'{iid}.png'
    mask_img = cv2.imread(str(png), cv2.IMREAD_GRAYSCALE)
    if mask_img is None:       
        return None, [], ann_id_start

    img = cv2.imread(str(jpg))
    if img is None:
        return None, [], ann_id_start
    h, w = img.shape[:2]
    img_info = dict(
        id=img_id,
        file_name=f'{iid}.jpg',
        width=w,
        height=h
    )

    ann_infos = []
    xml = ET.parse(str(voc_root / f'VOC{year}' / 'Annotations' / f'{iid}.xml')).getroot()
    for obj in xml.findall('object'):
        cls = obj.find('name').text.lower().strip()
        if cls not in CLASSES:
            continue
        cat_id = CLASSES.index(cls) + 1

        bnd = obj.find('bndbox')
        xmin = int(float(bnd.find('xmin').text))
        ymin = int(float(bnd.find('ymin').text))
        xmax = int(float(bnd.find('xmax').text))
        ymax = int(float(bnd.find('ymax').text))
        xmin, ymin = max(0, xmin), max(0, ymin)
        xmax, ymax = min(w, xmax), min(h, ymax)
        if xmin >= xmax or ymin >= ymax:
            continue

        roi = mask_img[ymin:ymax, xmin:xmax]
        valid = roi[(roi != 0) & (roi != 255)]
        if valid.size == 0:
            continue

        instance_id = np.bincount(valid.flatten()).argmax()
        rle = mask_from_img(mask_img, instance_id)
        if rle is None:
            continue
        area = mask_utils.area(rle).item()
        bbox = mask_utils.toBbox(rle).tolist()

        ann_infos.append(dict(
            id=ann_id_start,
            image_id=img_id,
            category_id=cat_id,
            segmentation=rle,
            area=area,
            bbox=bbox,
            iscrowd=0
        ))
        ann_id_start += 1
    if not ann_infos:
        return None, [], ann_id_start
    return img_info, ann_infos, ann_id_start


def build_coco_struct():
    return {
        'info': dict(version='1.0'),
        'licenses': [],
        'categories': [dict(id=i + 1, name=n) for i, n in enumerate(CLASSES)],
        'images': [],
        'annotations': []
    }


def main():
    args = parse_args()
    voc_root = Path(args.voc_root).expanduser().resolve()
    out_root = Path(args.out).expanduser().resolve()
    (out_root / 'annotations').mkdir(parents=True, exist_ok=True)
    (out_root / 'train0712').mkdir(exist_ok=True)
    (out_root / 'val07').mkdir(exist_ok=True)

    train_coco = build_coco_struct()
    val_coco = build_coco_struct()
    img_id = ann_id = 1
    val_img_id = val_ann_id = 1

    for year, splits in SPLIT_PLAN.items():
        for sp in splits:
            ids = iter_split_ids(voc_root, year, sp)
            for iid in tqdm.tqdm(ids, desc=f'VOC{year}-{sp}'):
                res = process_image(voc_root, year, iid, img_id, ann_id)
                if res is None:
                    continue
                img_info, ann_list, ann_id = res
                train_coco['images'].append(img_info)
                train_coco['annotations'].extend(ann_list)

                src = voc_root / f'VOC{year}' / 'JPEGImages' / img_info['file_name']
                dst = out_root / 'train0712' / img_info['file_name']
                if not dst.exists():
                    os.symlink(src.resolve(), dst)
                img_id += 1

    year_val, split_val = VAL_SPLIT
    for iid in tqdm.tqdm(iter_split_ids(voc_root, year_val, split_val),
                         desc=f'VOC{year_val}-{split_val} (val)'):
        res = process_image(voc_root, year_val, iid, val_img_id, val_ann_id)
        if res is None:
            continue
        img_info, ann_list, val_ann_id = res
        val_coco['images'].append(img_info)
        val_coco['annotations'].extend(ann_list)

        src = voc_root / f'VOC{year_val}' / 'JPEGImages' / img_info['file_name']
        dst = out_root / 'val07' / img_info['file_name']
        if not dst.exists():
            os.symlink(src.resolve(), dst)
        val_img_id += 1

    ann_dir = out_root / 'annotations'
    json.dump(train_coco, open(ann_dir / 'instances_train0712.json', 'w'))
    json.dump(val_coco,   open(ann_dir / 'instances_val07.json',   'w'))
    print(f'✓ instances_train0712.json  images={len(train_coco["images"])}')
    print(f'✓ instances_val07.json      images={len(val_coco["images"])}')


if __name__ == '__main__':
    main()
