"""
Mask R-CNN 可视化

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
        --ckpt ./work_dirs/mask_rcnn_r50_fpn_voc_ins/best_coco_bbox_mAP_epoch_147.pth\
        --score_thr 0.3

    python visualize_voc_results.py --mode external \
        --cfg ./work_dirs/mask_rcnn_r50_fpn_voc_ins/merged_cfg.py \
        --ckpt ./work_dirs/mask_rcnn_r50_fpn_voc_ins/best_coco_bbox_mAP_epoch_147.pth \
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
from mmengine.config import Config, ConfigDict
from mmengine import mkdir_or_exist
from mmengine.registry import init_default_scope
from mmdet.apis import init_detector, inference_detector
from mmdet.registry import VISUALIZERS
from mmdet.structures import DetDataSample
from mmengine.dataset import Compose 
from mmengine.structures import InstanceData


init_default_scope('mmdet')

def setup_model(cfg_path, ckpt_path, device='cuda:0'):
    cfg = Config.fromfile(cfg_path)
    print(f"--- MMDetection Version: 3.3.0 ---")

    if not cfg.model.get('test_cfg', {}).get('rpn'):
        print(f"--- 致命错误: cfg.model.test_cfg.rpn 在配置文件中缺失或为空! 请修正 merged_cfg.py。")
    else:
        cfg.model.test_cfg.rpn['output_proposals'] = True
        if 'save_best_proposals' in cfg.model.test_cfg.rpn: 
            del cfg.model.test_cfg.rpn['save_best_proposals']
        print(f"--- 调试 (MMDet 3.x): 确保 cfg.model.test_cfg.rpn 存在并设置了 output_proposals: {cfg.model.test_cfg.rpn}")

    if hasattr(cfg.model, 'rpn_head'):
        if not hasattr(cfg.model.rpn_head, 'test_cfg'):
            cfg.model.rpn_head.test_cfg = ConfigDict()
            print(f"--- 调试 (MMDet 3.x): 已为 cfg.model.rpn_head 创建空的 'test_cfg'")

        if cfg.model.get('test_cfg', {}).get('rpn'):
            for key, value in cfg.model.test_cfg.rpn.items():
                if key not in cfg.model.rpn_head.test_cfg:
                    cfg.model.rpn_head.test_cfg[key] = value

        cfg.model.rpn_head.test_cfg['output_proposals'] = True 
        if 'save_best_proposals' in cfg.model.rpn_head.test_cfg: 
            del cfg.model.rpn_head.test_cfg['save_best_proposals']
        print(f"--- 调试 (MMDet 3.x): 最终的 cfg.model.rpn_head.test_cfg: {cfg.model.rpn_head.test_cfg}")
    else:
        print("--- 警告 (MMDet 3.x): cfg.model.rpn_head 未找到。")

    if hasattr(cfg.model, 'roi_head') and cfg.model.get('test_cfg', {}).get('rcnn'):
        if not hasattr(cfg.model.roi_head, 'test_cfg'):
            cfg.model.roi_head.test_cfg = ConfigDict()
        for key, value in cfg.model.test_cfg.rcnn.items(): # 从 model.test_cfg.rcnn 获取参数
             if key not in cfg.model.roi_head.test_cfg: # 避免覆盖 roi_head 中可能已有的更精确的配置
                cfg.model.roi_head.test_cfg[key] = value
        print(f"--- 调试 (MMDet 3.x): 更新/设置 cfg.model.roi_head.test_cfg: {cfg.model.roi_head.test_cfg}")


    model = init_detector(cfg, ckpt_path, device=device)
    model.eval()
    # 新增的运行时检查
    if hasattr(model, 'rpn_head') and hasattr(model.rpn_head, 'test_cfg'):
        print(f"--- 运行时检查: model.rpn_head.test_cfg: {model.rpn_head.test_cfg}")
    else:
        print(f"--- 运行时检查: model.rpn_head 或其 test_cfg 未找到。")

    print(f"模型已初始化。使用的数据预处理器: {type(model.data_preprocessor)}")
    return model, cfg


def visualize(mode, model, cfg, images, out_root, score_thr):
    mkdir_or_exist(out_root)
    visualizer_cfg = dict(
        type='DetLocalVisualizer',
        name='vis',
        save_dir=out_root,
        line_width=2,          
        bbox_color=(0, 255, 0), 
        text_color=(200, 200, 200) 
    )
    visualizer = VISUALIZERS.build(visualizer_cfg)

    if hasattr(visualizer, 'draw_proposals'): 
        print(f"--- 调试: visualizer.draw_proposals 属性存在，值为: {visualizer.draw_proposals}")
    else:
        print(f"--- 调试: visualizer 没有 draw_proposals 属性 (MMDet 3.x 正常行为)")


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
        dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'pad_shape', 'batch_input_shape')) # 确保 pad_shape, batch_input_shape 也在这里
    ]
    composed_pipeline = Compose(simple_test_pipeline_cfg)
    out_root_abs = os.path.abspath(os.path.join(out_root, visualizer_cfg['name'])) 
    mkdir_or_exist(out_root_abs)


    for img_path in images:
        print(f"\n--- VISUALIZE LOOP: Current img_path = {img_path} ---")
        raw_img_bgr = cv2.imread(img_path)
        if raw_img_bgr is None:
            print(f"错误: 无法读取图片 {img_path}。跳过此图片。")
            continue
        
        img_for_visualizer_rgb = cv2.cvtColor(raw_img_bgr, cv2.COLOR_BGR2RGB)
        current_img_prefix = os.path.splitext(os.path.basename(img_path))[0]

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

        pipeline_input_data = dict(img_path=img_path, ori_shape=raw_img_bgr.shape[:2]) 
        processed_data = composed_pipeline(pipeline_input_data)
        batched_data_for_model = {'inputs': [processed_data['inputs']], 'data_samples': [processed_data['data_samples']]}
        
        with torch.no_grad():
            data_for_predict = model.data_preprocessor(batched_data_for_model, training=False)
            result_datasample_list = model.predict(data_for_predict['inputs'], data_for_predict['data_samples'])
        
        ds = result_datasample_list[0]
        print(f"====== 模型推理结果 for {os.path.basename(img_path)} ======")
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
                
                proposals_at_input_scale = ds.proposals.clone()

                num_proposals_to_draw = 20
                if hasattr(proposals_at_input_scale, 'scores') and \
                   proposals_at_input_scale.scores is not None and \
                   isinstance(proposals_at_input_scale.scores, torch.Tensor) and \
                   proposals_at_input_scale.scores.ndim == 1 and \
                   len(proposals_at_input_scale.scores) > 0 and \
                   len(proposals_at_input_scale.scores) == len(proposals_at_input_scale.bboxes):
                    
                    print(f"--- 调试: 原始 proposals 数量: {len(proposals_at_input_scale.scores)}")
                    sorted_inds = torch.argsort(proposals_at_input_scale.scores, descending=True)
                    top_k_inds = sorted_inds[:num_proposals_to_draw]
                    
                    temp_inst_data = InstanceData() 
                    temp_inst_data.bboxes = proposals_at_input_scale.bboxes[top_k_inds]
                    temp_inst_data.scores = proposals_at_input_scale.scores[top_k_inds]
                    if hasattr(proposals_at_input_scale, 'labels'): 
                        temp_inst_data.labels = proposals_at_input_scale.labels[top_k_inds]
                    else:
                        temp_inst_data.labels = torch.zeros(len(top_k_inds), dtype=torch.long, device=temp_inst_data.bboxes.device)
                    proposals_at_input_scale = temp_inst_data
                    print(f"--- 调试: 将可视化分数最高的 {len(proposals_at_input_scale.bboxes)} 个 proposals。")
                else: 
                    num_to_take = min(num_proposals_to_draw, len(proposals_at_input_scale.bboxes))
                    proposals_at_input_scale = proposals_at_input_scale[:num_to_take]
                    print(f"--- 调试: proposals 分数信息不足，可视化前 {num_to_take} 个 proposals。")
            
                print(f"\n--- OpenCV 手动绘制调试 ---")
                img_for_opencv_draw = raw_img_bgr.copy() 
                
                ds_ori_shape = ds.get('ori_shape')       
                ds_scale_factor = ds.get('scale_factor') 

                if ds_ori_shape and ds_scale_factor and hasattr(proposals_at_input_scale, 'bboxes') and len(proposals_at_input_scale.bboxes) > 0:
                    bboxes_np_scaled = proposals_at_input_scale.bboxes.clone().cpu().numpy()
                    
                    rescaled_bboxes_np = bboxes_np_scaled.copy()
                    rescaled_bboxes_np[:, 0::2] /= ds_scale_factor[0] # x1, x2
                    rescaled_bboxes_np[:, 1::2] /= ds_scale_factor[1] # y1, y2

                    print(f"    OpenCV: 待绘制的 rescaled bboxes (前5个, 应为原始图像坐标系): \n{rescaled_bboxes_np[:5]}")
                    print(f"    OpenCV: 原始图像尺寸 ori_shape (H, W): {ds_ori_shape}")

                    num_drawn_opencv = 0
                    for i in range(len(rescaled_bboxes_np)):
                        x1, y1, x2, y2 = rescaled_bboxes_np[i]
                        x1_c, y1_c, x2_c, y2_c = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
                        
                        x1_c = max(0, x1_c)
                        y1_c = max(0, y1_c)
                        x2_c = min(img_for_opencv_draw.shape[1] - 1, x2_c) # img_for_opencv_draw.shape[1] is width
                        y2_c = min(img_for_opencv_draw.shape[0] - 1, y2_c) # img_for_opencv_draw.shape[0] is height

                        if x1_c < x2_c and y1_c < y2_c:
                            cv2.rectangle(img_for_opencv_draw, (x1_c, y1_c), (x2_c, y2_c), (0, 255, 0), 2) 
                            num_drawn_opencv +=1
                    
                    print(f"    OpenCV: 尝试绘制 {len(rescaled_bboxes_np)} 个框, 实际有效并绘制 {num_drawn_opencv} 个框。")
                    opencv_save_path = os.path.join(out_root_abs, f"{current_img_prefix}_proposal_opencv_direct_draw.png")
                    save_success = cv2.imwrite(opencv_save_path, img_for_opencv_draw)
                    if save_success:
                        print(f"    调试: 手动用 OpenCV 绘制的 proposal 图像已保存到: {opencv_save_path}")
                    else:
                        print(f"    错误: OpenCV 图像保存失败到: {opencv_save_path}")
                else:
                    print("--- OpenCV: 缺少元信息或无 proposals 可绘制，跳过 OpenCV 手动绘制。")

                print(f"\n--- MMDetection Visualizer 使用手动缩放坐标调试 ---")
                if ds_ori_shape and ds_scale_factor and hasattr(proposals_at_input_scale, 'bboxes') and len(proposals_at_input_scale.bboxes) > 0:
                    manually_rescaled_proposals = InstanceData()
                    manually_rescaled_proposals.bboxes = torch.from_numpy(rescaled_bboxes_np).to(proposals_at_input_scale.bboxes.device)
                    if hasattr(proposals_at_input_scale, 'scores'):
                        manually_rescaled_proposals.scores = proposals_at_input_scale.scores.clone()
                    if hasattr(proposals_at_input_scale, 'labels'):
                        manually_rescaled_proposals.labels = proposals_at_input_scale.labels.clone()
                    else: 
                        manually_rescaled_proposals.labels = torch.zeros(len(manually_rescaled_proposals.bboxes), dtype=torch.long, device=manually_rescaled_proposals.bboxes.device)


                    pd_for_mmdet_vis = DetDataSample()
                    pd_for_mmdet_vis.pred_instances = manually_rescaled_proposals
                    meta_for_mmdet_vis = {}
                    meta_for_mmdet_vis['ori_shape'] = ds_ori_shape
                    meta_for_mmdet_vis['img_shape'] = ds_ori_shape 
                    meta_for_mmdet_vis['scale_factor'] = (1.0, 1.0) 
                    meta_for_mmdet_vis['pad_shape'] = ds_ori_shape 
                    if ds.get('img_path'): meta_for_mmdet_vis['img_path'] = ds.get('img_path')
                    pd_for_mmdet_vis.set_metainfo(meta_for_mmdet_vis)
                    
                    print(f"    MMDet Vis: pd_for_mmdet_vis 的元信息:")
                    for key_meta in ['ori_shape', 'img_shape', 'scale_factor', 'pad_shape']:
                        print(f"      pd_for_mmdet_vis.{key_meta}: {pd_for_mmdet_vis.get(key_meta, '未设置')}")
                    if hasattr(pd_for_mmdet_vis.pred_instances, 'bboxes'):
                        print(f"    MMDet Vis: 待绘制的 bboxes (前5个, 应为原始图像坐标系): {pd_for_mmdet_vis.pred_instances.bboxes[:5]}")


                    mmdet_vis_save_name = f'{current_img_prefix}_proposal_mmdet_scaled_coords'
                    visualizer.add_datasample(
                        name=mmdet_vis_save_name,
                        image=img_for_visualizer_rgb,
                        data_sample=pd_for_mmdet_vis,
                        draw_gt=False,
                        draw_pred=True, 
                        show=False,
                        pred_score_thr=0.0, 
                    )
                    mmdet_vis_out_path_manual = os.path.join(out_root_abs, f"{mmdet_vis_save_name}_manual_save.png")
                    try:
                        drawn_mmdet_img = visualizer.get_image()
                        if drawn_mmdet_img is not None:
                            cv2.imwrite(mmdet_vis_out_path_manual, cv2.cvtColor(np.ascontiguousarray(drawn_mmdet_img), cv2.COLOR_RGB2BGR))
                            print(f"    调试: MMDetection Visualizer (手动缩放坐标) 的 proposal 图像已保存到: {mmdet_vis_out_path_manual}")
                        else: print(f"    错误: MMDetection Visualizer (手动缩放坐标).get_image() 返回 None。")
                    except Exception as e_get_img_mmdet: print(f"    错误: MMDetection Visualizer (手动缩放坐标) 保存时发生异常: {e_get_img_mmdet}")
                else:
                    print("--- MMDetection Visualizer: 缺少元信息或无 proposals 可进行手动缩放绘制，跳过。")

            else: 
                print(f"  跳过 {current_img_prefix} 的提议框可视化：无有效提议框数据。")

        # 2) Final bbox 和 3) Instance mask 可视化
        if hasattr(ds, 'pred_instances') and ds.pred_instances is not None and \
           hasattr(ds.pred_instances, 'bboxes') and len(ds.pred_instances.bboxes) > 0:
               
            print(f"  为 {current_img_prefix}_bbox 添加最终边界框可视化 (score_thr={score_thr})...")
            ds_for_bbox_visualization = DetDataSample()
            pred_instances_bbox_only = InstanceData()
            if hasattr(ds.pred_instances, 'bboxes'):
                pred_instances_bbox_only.bboxes = ds.pred_instances.bboxes
            if hasattr(ds.pred_instances, 'scores'):
                pred_instances_bbox_only.scores = ds.pred_instances.scores
            if hasattr(ds.pred_instances, 'labels'):
                pred_instances_bbox_only.labels = ds.pred_instances.labels
            ds_for_bbox_visualization.pred_instances = pred_instances_bbox_only
            
            if hasattr(ds, 'meta') and isinstance(ds.meta, dict):
                ds_for_bbox_visualization.set_metainfo(ds.meta)
            else:
                print(f"    警告: ds.meta 不是预期的字典或不存在。尝试手动为 ds_for_bbox_visualization 复制关键元信息...")
                metainfo_to_copy = {}
                keys_to_copy = ['img_path', 'ori_shape', 'img_shape', 'scale_factor', 'pad_shape', 'batch_input_shape']
                all_critical_keys_found_bbox = True
                critical_keys = ['ori_shape', 'img_shape', 'scale_factor']
                for key in keys_to_copy:
                    value_found = False
                    if hasattr(ds, key): value = getattr(ds, key); value_found = True
                    elif ds.get(key) is not None: value = ds.get(key); value_found = True
                    
                    if value_found and not isinstance(value, (DetDataSample, InstanceData)):
                        metainfo_to_copy[key] = value
                    elif key in critical_keys: all_critical_keys_found_bbox = False
                
                if metainfo_to_copy:
                    ds_for_bbox_visualization.set_metainfo(metainfo_to_copy)
                    print(f"      调试: 已将元信息设置到 ds_for_bbox_visualization。")
                    if not all_critical_keys_found_bbox:
                         print(f"      严重警告: ds_for_bbox_visualization 未能复制所有关键元信息。")
                else:
                    print(f"      严重警告: 未能从 ds 为 ds_for_bbox_visualization 复制任何元信息。")

            original_show_mask_setting_bbox = True 
            if hasattr(visualizer, 'show_mask'): original_show_mask_setting_bbox = visualizer.show_mask
            visualizer.show_mask = False
            print(f"    调试: 为 bbox 可视化，临时设置 visualizer.show_mask = {visualizer.show_mask}")

            visualizer.add_datasample(
                name=f'{current_img_prefix}_bbox', 
                image=img_for_visualizer_rgb,
                data_sample=ds_for_bbox_visualization,
                draw_gt=False, 
                draw_pred=True,
                pred_score_thr=score_thr, 
                show=False
            )
  
            if not os.path.exists(expected_bbox_img_file): # 或者您有一个 FORCE_MANUAL_SAVE 标志
                print(f"    调试: BBOX 图像文件 '{expected_bbox_img_file}' 未通过 add_datasample 自动创建。")
                print(f"    尝试从 visualizer 手动获取并保存图像 {current_img_prefix}_bbox ...")
                try:
                    drawn_bbox_image_rgb = visualizer.get_image() 
                    if drawn_bbox_image_rgb is not None:
                        print(f"      调试: visualizer.get_image() (for bbox) 成功返回图像，形状: {drawn_bbox_image_rgb.shape}, 类型: {drawn_bbox_image_rgb.dtype}")
                        manual_save_path_bbox_png = os.path.join(out_root_abs, f"{current_img_prefix}_bbox_manual_save.png")
                        success_manual_save_bbox_png = cv2.imwrite(manual_save_path_bbox_png, cv2.cvtColor(np.ascontiguousarray(drawn_bbox_image_rgb), cv2.COLOR_RGB2BGR))
                        if success_manual_save_bbox_png and os.path.exists(manual_save_path_bbox_png):
                            print(f"      调试: 手动从 visualizer.get_image() (for bbox) 保存图像为 PNG 成功: {manual_save_path_bbox_png}")
                        else:
                            print(f"      调试: 手动从 visualizer.get_image() (for bbox) 保存图像为 PNG 失败。")
                    else:
                        print(f"      调试: visualizer.get_image() (for bbox) 返回了 None，无法手动保存。")
                except Exception as e_get_img_bbox:
                    print(f"      调试: 调用 visualizer.get_image() (for bbox) 或手动保存 BBOX 图像时出错: {e_get_img_bbox}")
                    import traceback
                    traceback.print_exc()
            elif os.path.exists(expected_bbox_img_file):
                print(f"    调试: BBOX 图像文件已通过 add_datasample 自动创建或之前已手动保存: {expected_bbox_img_file}")

            if hasattr(visualizer, 'show_mask'):
                visualizer.show_mask = original_show_mask_setting_bbox
            print(f"    调试: 恢复 visualizer.show_mask = {visualizer.show_mask}")
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

    if args.mode == 'test':
        images = [
            '/mnt/data/jichuan/openmmlab_voc_project/data/voc_ins/val07/002954.jpg',
            '/mnt/data/jichuan/openmmlab_voc_project/data/voc_ins/val07/000676.jpg',
            '/mnt/data/jichuan/openmmlab_voc_project/data/voc_ins/val07/000799.jpg',
            '/mnt/data/jichuan/openmmlab_voc_project/data/voc_ins/val07/009841.jpg',
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
    model, cfg = setup_model(args.cfg, args.ckpt)
    
    print("\n--- 开始 DetInferencer 测试 ---")
    from mmdet.apis import DetInferencer
    try:
        inferencer_device = cfg.get('device', 'cuda:0' if torch.cuda.is_available() else 'cpu')
        inferencer = DetInferencer(model=args.cfg, weights=args.ckpt, device=inferencer_device)

        if images:
            test_image_for_inferencer = images[0]
            print(f"DetInferencer 正在处理图片: {test_image_for_inferencer}")
            result_inferencer = inferencer(test_image_for_inferencer, return_datasamples=True)
            
            if result_inferencer and 'predictions' in result_inferencer and result_inferencer['predictions']:
                ds_from_inferencer = result_inferencer['predictions'][0]
                
                print(f"DetInferencer 返回的 DetDataSample 类型: {type(ds_from_inferencer)}")
                if hasattr(ds_from_inferencer, 'proposals') and ds_from_inferencer.proposals is not None and len(ds_from_inferencer.proposals) > 0 :
                    print(f"成功: DetInferencer 返回的 DetDataSample 中包含 proposals！")
                    if hasattr(ds_from_inferencer.proposals, 'bboxes'):
                        print(f"  Proposals 数量: {len(ds_from_inferencer.proposals.bboxes)}")
                        if len(ds_from_inferencer.proposals.bboxes) > 0:
                             print(f"  第一个 proposal bboxes (前5个值): {ds_from_inferencer.proposals.bboxes[0][:5]}")
                             if hasattr(ds_from_inferencer.proposals, 'scores') and \
                                ds_from_inferencer.proposals.scores is not None and \
                                isinstance(ds_from_inferencer.proposals.scores, torch.Tensor) and \
                                ds_from_inferencer.proposals.scores.numel() > 0: # 使用 numel() 检查是否有元素
                                 print(f"  Proposal scores (前5个): {ds_from_inferencer.proposals.scores[:min(5, len(ds_from_inferencer.proposals.scores))]}")
                             elif hasattr(ds_from_inferencer.proposals, 'scores') and ds_from_inferencer.proposals.scores is not None:
                                 print(f"  Proposal scores 内容 (可能为空或非预期格式): {ds_from_inferencer.proposals.scores}")
                             else:
                                 print(f"  ds_from_inferencer.proposals 中没有有效的 'scores'。")
                    else:
                        print(f"  ds_from_inferencer.proposals 中没有 'bboxes' 属性。Proposals 内容: {ds_from_inferencer.proposals}")
                else:
                    print(f"失败: DetInferencer 返回的 DetDataSample 中仍然没有 proposals 或 proposals 为空。")
                
                if hasattr(ds_from_inferencer, 'pred_instances') and ds_from_inferencer.pred_instances is not None:
                    print(f"  DetInferencer pred_instances 数量: {len(ds_from_inferencer.pred_instances.bboxes)}")
            else:
                print("DetInferencer 未返回有效的预测结果。")
        else:
            print("图片列表为空，无法进行 DetInferencer 测试。")

    except Exception as e_inferencer:
        print(f"DetInferencer 测试过程中发生错误: {e_inferencer}")
        import traceback
        traceback.print_exc()
    print("--- DetInferencer 测试结束 ---\n")
    
    visualize(args.mode, model, cfg, images, out_root, args.score_thr)

if __name__ == '__main__':
    main()
