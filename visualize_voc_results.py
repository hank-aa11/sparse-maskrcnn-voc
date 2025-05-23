"""
Mask R-CNN 可视化脚本

功能：
1) 测试模式（test）：
    - 对 VOC 测试集 4 张图片可视化：
      * RPN Proposal boxes (如果模型输出中包含)
      * 最终预测 bbox
      * 实例分割 mask
2) 外部模式（external）：
    - 对 3 张外部图片可视化同一模型的预测：
      * bbox、instance mask、类别标签、得分

用法示例：
    python visualize_voc_results.py --mode test \
        --cfg ./work_dirs/mask_rcnn_r50_fpn_voc_ins/merged_cfg.py \
        --ckpt ./work_dirs/mask_rcnn_r50_fpn_voc_ins/best_coco_bbox_mAP_epoch_35.pth\
        --score_thr 0.3

    python visualize_voc_results.py --mode external \
        --cfg ./work_dirs/mask_rcnn_r50_fpn_voc_ins/merged_cfg.py \
        --ckpt ./work_dirs/mask_rcnn_r50_fpn_voc_ins/best_coco_bbox_mAP_epoch_35.pth \
        --score_thr 0.3

可视化结果保存在 (注意会有一个 'vis' 子目录):
    test    -> ./vis/test_set_vis/vis/
    external-> ./vis/external_vis/vis/
"""
import os
import argparse
import cv2
import torch
import numpy as np
from mmengine.config import Config
from mmengine import mkdir_or_exist
from mmengine.registry import init_default_scope
from mmdet.apis import init_detector
from mmdet.registry import VISUALIZERS
from mmdet.structures import DetDataSample
from mmengine.dataset import Compose 


init_default_scope('mmdet')

def setup_model(cfg_path, ckpt_path, device='cuda:0'):
    cfg = Config.fromfile(cfg_path)
    model = init_detector(cfg, ckpt_path, device=device) 
    model.eval()
    print(f"模型已初始化。使用的数据预处理器: {type(model.data_preprocessor)}")
    return model, cfg

def visualize(mode, model, cfg, images, out_root, score_thr):
    mkdir_or_exist(out_root)
    visualizer_cfg = dict(type='DetLocalVisualizer', name='vis', save_dir=out_root)
    visualizer = VISUALIZERS.build(visualizer_cfg)
    
    if cfg.get('metainfo', None) is not None:
        visualizer.dataset_meta = cfg.metainfo
    elif cfg.get('train_dataloader', None) is not None and \
         cfg.train_dataloader.get('dataset', None) is not None and \
         cfg.train_dataloader.dataset.get('metainfo', None) is not None:
        visualizer.dataset_meta = cfg.train_dataloader.dataset.metainfo
        print("从 train_dataloader.dataset.metainfo 获取 metainfo 用于可视化。")
    else:
        print("警告: 配置文件中找不到 metainfo，可视化结果可能缺少类别名称。")

    simple_test_pipeline_cfg = [
        dict(type='LoadImageFromFile', backend_args=None),
        dict(type='Resize', scale=(1333, 800), keep_ratio=True),
        dict(type='Pad', size_divisor=32),
        dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
    ]
    composed_pipeline = Compose(simple_test_pipeline_cfg)

    for img_path in images:
        print(f"正在处理 {img_path}...")
        raw_img_bgr = cv2.imread(img_path)
        if raw_img_bgr is None:
            print(f"错误: 无法读取图片 {img_path}。跳过此图片。")
            continue
        
        img_for_visualizer_rgb = cv2.cvtColor(raw_img_bgr, cv2.COLOR_BGR2RGB)
        current_img_prefix = os.path.splitext(os.path.basename(img_path))[0]

        # ===================== 保存原始加载的图像 =====================
        debug_raw_save_dir = os.path.join(out_root, visualizer_cfg['name'])
        raw_save_path = os.path.join(out_root, visualizer_cfg['name'], f"{current_img_prefix}_original_loaded.png")
        
        print(f"  调试: 尝试手动保存原始加载的图像 (BGR格式) 到 {raw_save_path} ...")
        try:
            mkdir_or_exist(os.path.join(out_root, visualizer_cfg['name']))
            success_save_raw = cv2.imwrite(raw_save_path, raw_img_bgr)
            if success_save_raw and os.path.exists(raw_save_path):
                print(f"  调试: 手动保存原始加载的图像成功: {raw_save_path}")
            else:
                print(f"  调试: 手动保存原始加载的图像失败。cv2.imwrite 返回 {success_save_raw}")
        except Exception as e_raw_save:
            print(f"  调试: 手动保存原始加载的图像时出错: {e_raw_save}")
            import traceback
            traceback.print_exc()
        # =====================================================================

        pipeline_input_data = dict(img_path=img_path)
        try:
            processed_data = composed_pipeline(pipeline_input_data)
        except Exception as e:
            print(f"错误: 执行图像处理流水线失败 for {img_path}: {e}")
            import traceback
            traceback.print_exc()
            continue

        batched_data_for_model = {'inputs': [processed_data['inputs']], 'data_samples': [processed_data['data_samples']]}
        
        try:
            with torch.no_grad():
                result_datasample_list = model.test_step(batched_data_for_model)
        except Exception as e:
            print(f"错误: 模型推理 (model.test_step) 失败 for {img_path}: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        print(f"====== 模型推理结果 for {os.path.basename(img_path)} ======")
        if not result_datasample_list:
            print("错误: model.test_step 返回了空的结果列表！")
            continue 

        ds = result_datasample_list[0]
        
        if ds is None or not isinstance(ds, DetDataSample):
            print("错误: DetDataSample 为空或类型不正确！")
            print(f"  model.test_step 返回的 result_datasample_list 内容: {result_datasample_list}")
            continue

        print(f"DetDataSample (ds) 内容简述: {ds}")
        if hasattr(ds, 'proposals') and ds.proposals is not None:
            print(f"  Proposals (提议框):")
            if hasattr(ds.proposals, 'bboxes') and len(ds.proposals.bboxes) > 0:
                print(f"    - 数量: {len(ds.proposals.bboxes)}")
            else:
                print(f"    - 数量: 0 或 proposals.bboxes 不存在")
        else:
            print("  Proposals (提议框): ds 中不存在或为 None")

        if hasattr(ds, 'pred_instances') and ds.pred_instances is not None:
            print(f"  Predicted Instances (预测实例):")
            if hasattr(ds.pred_instances, 'bboxes') and len(ds.pred_instances.bboxes) > 0:
                print(f"    - 数量: {len(ds.pred_instances.bboxes)}")
                if hasattr(ds.pred_instances, 'scores'): print(f"    - 置信度 (前5个): {ds.pred_instances.scores[:5]}")
                if hasattr(ds.pred_instances, 'labels'): print(f"    - 标签 (前5个): {ds.pred_instances.labels[:5]}")
            else:
                print(f"    - 数量: 0 或 pred_instances.bboxes 不存在")
        else:
            print("  Predicted Instances (预测实例): ds 中不存在或为 None")
        print("==========================================")
        
        # 1) Proposal 可视化
        if mode == 'test':
            if hasattr(ds, 'proposals') and ds.proposals is not None and \
               hasattr(ds.proposals, 'bboxes') and len(ds.proposals.bboxes) > 0:
                pd_proposal = DetDataSample()
                pd_proposal.proposals = ds.proposals.clone()
                print(f"  为 {current_img_prefix}_proposal 添加提议框可视化...")
                visualizer.add_datasample(
                    name=f'{current_img_prefix}_proposal', image=img_for_visualizer_rgb,
                    data_sample=pd_proposal, draw_gt=False, draw_pred=False,
                    draw_proposal=True, show=False
                )
                expected_prop_img_file = os.path.join(out_root, visualizer_cfg['name'], f"{current_img_prefix}_proposal.png")
                if os.path.exists(expected_prop_img_file):
                    print(f"  调试: PROPOSAL 图像文件已创建: {expected_prop_img_file}")
                else:
                    print(f"  调试: PROPOSAL 图像文件未创建: {expected_prop_img_file}")
            else:
                print(f"  跳过 {current_img_prefix} 的提议框可视化：无有效提议框数据。")

        # 2) Final bbox 和 3) Instance mask 可视化
        if hasattr(ds, 'pred_instances') and ds.pred_instances is not None and \
           hasattr(ds.pred_instances, 'bboxes') and len(ds.pred_instances.bboxes) > 0:
            
            print(f"  为 {current_img_prefix}_bbox 添加最终边界框可视化 (score_thr={score_thr})...")
            visualizer.add_datasample(
                name=f'{current_img_prefix}_bbox', image=img_for_visualizer_rgb,
                data_sample=ds, draw_gt=False, draw_pred=True,
                pred_score_thr=score_thr, show=False
            )
            expected_bbox_img_file = os.path.join(out_root, visualizer_cfg['name'], f"{current_img_prefix}_bbox.png")
            if os.path.exists(expected_bbox_img_file):
                print(f"  调试: BBOX 图像文件已通过 visualizer.add_datasample 创建: {expected_bbox_img_file}")
            else:
                print(f"  调试: BBOX 图像文件未通过 visualizer.add_datasample 创建: {expected_bbox_img_file}")
                print(f"    尝试从 visualizer 手动获取并保存图像 {current_img_prefix}_bbox ...")
                try:
                    drawn_image_rgb_internal = visualizer.get_image() 
                    if drawn_image_rgb_internal is not None:
                        print(f"    调试: visualizer.get_image() 成功返回图像，形状: {drawn_image_rgb_internal.shape}, 类型: {drawn_image_rgb_internal.dtype}")
                        drawn_image_rgb_internal_contiguous = np.ascontiguousarray(drawn_image_rgb_internal)
                        drawn_image_bgr_internal = cv2.cvtColor(drawn_image_rgb_internal_contiguous, cv2.COLOR_RGB2BGR)
                        
                        manual_save_path_png = os.path.join(out_root, visualizer_cfg['name'], f"{current_img_prefix}_bbox_manual_save.png")
                        success_manual_save_png = cv2.imwrite(manual_save_path_png, drawn_image_bgr_internal)
                        
                        if success_manual_save_png and os.path.exists(manual_save_path_png):
                            print(f"    调试: 手动从 visualizer.get_image() 保存图像为 PNG 成功: {manual_save_path_png}")
                        else:
                            print(f"    调试: 手动从 visualizer.get_image() 保存图像为 PNG 失败。cv2.imwrite (PNG) 返回 {success_manual_save_png}")
                            manual_save_path_jpg = os.path.join(out_root, visualizer_cfg['name'], f"{current_img_prefix}_bbox_manual_save.jpg")
                            print(f"      尝试将图像保存为 JPG 格式到: {manual_save_path_jpg}")
                            success_manual_save_jpg = cv2.imwrite(manual_save_path_jpg, drawn_image_bgr_internal, [cv2.IMWRITE_JPEG_QUALITY, 95])
                            if success_manual_save_jpg and os.path.exists(manual_save_path_jpg):
                                print(f"      调试: 手动从 visualizer.get_image() 保存图像为 JPG 成功: {manual_save_path_jpg}")
                            else:
                                print(f"      调试: 手动从 visualizer.get_image() 保存图像为 JPG 也失败了。cv2.imwrite (JPG) 返回 {success_manual_save_jpg}")
                    else:
                        print(f"    调试: visualizer.get_image() 返回了 None，无法手动保存。")
                except Exception as e_get_img:
                    print(f"    调试: 调用 visualizer.get_image() 或手动保存 BBOX 图像时出错: {e_get_img}")
                    import traceback
                    traceback.print_exc()
            
            if hasattr(ds.pred_instances, 'masks') and ds.pred_instances.masks is not None:
                print(f"  为 {current_img_prefix}_mask 添加实例分割掩码可视化 (score_thr={score_thr})...")
                visualizer.add_datasample(
                    name=f'{current_img_prefix}_mask', image=img_for_visualizer_rgb,
                    data_sample=ds, draw_gt=False, draw_pred=True,
                    pred_score_thr=score_thr, show=False
                )
                expected_mask_img_file = os.path.join(out_root, visualizer_cfg['name'], f"{current_img_prefix}_mask.png")
                if os.path.exists(expected_mask_img_file):
                    print(f"  调试: MASK 图像文件已通过 visualizer.add_datasample 创建: {expected_mask_img_file}")
                else:
                    print(f"  调试: MASK 图像文件未通过 visualizer.add_datasample 创建: {expected_mask_img_file}")
                    print(f"    尝试从 visualizer 手动获取并保存图像 {current_img_prefix}_mask ...")
                    try:
                        drawn_image_rgb_internal_mask = visualizer.get_image()
                        if drawn_image_rgb_internal_mask is not None:
                            print(f"    调试: visualizer.get_image() (for mask) 成功返回图像，形状: {drawn_image_rgb_internal_mask.shape}, 类型: {drawn_image_rgb_internal_mask.dtype}")
                            drawn_image_rgb_internal_mask_contiguous = np.ascontiguousarray(drawn_image_rgb_internal_mask)
                            drawn_image_bgr_internal_mask = cv2.cvtColor(drawn_image_rgb_internal_mask_contiguous, cv2.COLOR_RGB2BGR)
                            
                            manual_save_path_mask_png = os.path.join(out_root, visualizer_cfg['name'], f"{current_img_prefix}_mask_manual_save.png")
                            success_manual_save_mask_png = cv2.imwrite(manual_save_path_mask_png, drawn_image_bgr_internal_mask)
                            if success_manual_save_mask_png and os.path.exists(manual_save_path_mask_png):
                                print(f"    调试: 手动从 visualizer.get_image() (for mask) 保存图像为 PNG 成功: {manual_save_path_mask_png}")
                            else:
                                print(f"    调试: 手动从 visualizer.get_image() (for mask) 保存图像为 PNG 失败。cv2.imwrite (PNG) 返回 {success_manual_save_mask_png}")
                                manual_save_path_mask_jpg = os.path.join(out_root, visualizer_cfg['name'], f"{current_img_prefix}_mask_manual_save.jpg")
                                print(f"      尝试将 MASK 图像保存为 JPG 格式到: {manual_save_path_mask_jpg}")
                                success_manual_save_mask_jpg = cv2.imwrite(manual_save_path_mask_jpg, drawn_image_bgr_internal_mask, [cv2.IMWRITE_JPEG_QUALITY, 95])
                                if success_manual_save_mask_jpg and os.path.exists(manual_save_path_mask_jpg):
                                    print(f"      调试: 手动从 visualizer.get_image() (for mask) 保存图像为 JPG 成功: {manual_save_path_mask_jpg}")
                                else:
                                    print(f"      调试: 手动从 visualizer.get_image() (for mask) 保存图像为 JPG 也失败了。cv2.imwrite (JPG) 返回 {success_manual_save_mask_jpg}")
                        else:
                            print(f"    调试: visualizer.get_image() (for mask) 返回了 None，无法手动保存。")
                    except Exception as e_get_img_mask:
                        print(f"    调试: 调用 visualizer.get_image() (for mask) 或手动保存 MASK 图像时出错: {e_get_img_mask}")
                        import traceback
                        traceback.print_exc()
            else:
                print(f"  跳过 {current_img_prefix} 的掩码可视化：pred_instances 中无掩码数据。")
        else:
            print(f"  跳过 {current_img_prefix} 的最终边界框和掩码可视化：无有效预测实例或边界框。")
            
    actual_save_dir = os.path.join(out_root, visualizer_cfg['name']) 
    print(f"{mode.capitalize()} 模式可视化处理完成。结果（如果生成了图像）应保存在: {actual_save_dir}")

def main():
    parser = argparse.ArgumentParser(description="Mask R-CNN VOC 结果可视化脚本")
    parser.add_argument('--mode', choices=['test', 'external'], required=True, help="运行模式: 'test' 或 'external'")
    parser.add_argument('--cfg', required=True, help="模型配置文件的路径 (.py)")
    parser.add_argument('--ckpt', required=True, help="模型权重文件的路径 (.pth)")
    parser.add_argument('--score_thr', type=float, default=0.3, help="显示预测结果的置信度阈值")
    args = parser.parse_args()

    model, cfg = setup_model(args.cfg, args.ckpt)
    
    if args.mode == 'test':
        images = [
            '/mnt/data/jichuan/openmmlab_voc_project/data/voc_ins/val07/000170.jpg',
            '/mnt/data/jichuan/openmmlab_voc_project/data/voc_ins/val07/000676.jpg',
            '/mnt/data/jichuan/openmmlab_voc_project/data/voc_ins/val07/000799.jpg',
            '/mnt/data/jichuan/openmmlab_voc_project/data/voc_ins/val07/008222.jpg',
        ]
        out_root = './vis/test_set_vis'
    else: 
        images = [
            '/mnt/data/jichuan/openmmlab_voc_project/extra_imgs/cat.jpg',
            '/mnt/data/jichuan/openmmlab_voc_project/extra_imgs/dog.jpg',
            '/mnt/data/jichuan/openmmlab_voc_project/extra_imgs/person.jpg',
        ]
        out_root = './vis/external_vis'
    
    if not images:
        print("错误：没有找到要处理的图片。请检查 'images' 列表。")
        return

    visualize(args.mode, model, cfg, images, out_root, args.score_thr)

if __name__ == '__main__':
    main()