#!/usr/bin/env python3
"""
HarvestLens AI - YOLOv8 Fruit Maturity Model Evaluation System
==============================================================
Automated, comprehensive evaluation script for the trained YOLOv8
model (`best.pt`) recognizing 32 fruit-ripeness classes.

Author: HarvestLens AI Team
License: Proprietary / Academic Research
"""

import argparse
import csv
import json
import os
import shutil
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import yaml
from ultralytics import YOLO

# 32 Canonical Classes (8 Fruits x 4 Maturity Stages)
ALL_CLASSES = [
    'Apple Overripe', 'Apple Ripe', 'Apple Rotten', 'Apple Unripe',
    'Banana Overripe', 'Banana Ripe', 'Banana Rotten', 'Banana Unripe',
    'Grape Overripe', 'Grape Ripe', 'Grape Rotten', 'Grape Unripe',
    'Mango Overripe', 'Mango Ripe', 'Mango Rotten', 'Mango Unripe',
    'Melon Overripe', 'Melon Ripe', 'Melon Rotten', 'Melon Unripe',
    'Orange Overripe', 'Orange Ripe', 'Orange Rotten', 'Orange Unripe',
    'Peach Overripe', 'Peach Ripe', 'Peach Rotten', 'Peach Unripe',
    'Pear Overripe', 'Pear Ripe', 'Pear Rotten', 'Pear Unripe'
]

FRUIT_TYPES = ['Apple', 'Banana', 'Grape', 'Mango', 'Melon', 'Orange', 'Peach', 'Pear']
MATURITY_STAGES = ['Unripe', 'Ripe', 'Overripe', 'Rotten']

STAGE_COLORS = {
    'Unripe': '#527A43',    # Leaf green
    'Ripe': '#E7B955',      # Golden ripe
    'Overripe': '#E0701A',  # Orange-brown
    'Rotten': '#7A1410'     # Dark brown-red
}

FRUIT_COLORS = {
    'Apple': '#D6303A',
    'Banana': '#FFD23F',
    'Grape': '#6B2F8F',
    'Mango': '#F5A623',
    'Melon': '#A8D08A',
    'Orange': '#FF9A1F',
    'Peach': '#FFB27A',
    'Pear': '#B99A3A'
}


def find_model_weights(requested_path=None):
    """Locate the trained YOLOv8 weights file."""
    candidates = []
    if requested_path:
        candidates.append(requested_path)
    
    candidates.extend([
        os.path.join(os.getcwd(), 'best.pt'),
        os.path.join(os.getcwd(), 'models', 'best.pt'),
        os.path.join(os.path.dirname(os.getcwd()), 'Trained model', 'best.pt'),
        os.path.join(os.path.dirname(os.getcwd()), 'best.pt')
    ])

    for path in candidates:
        if path and os.path.isfile(path):
            return os.path.abspath(path)
    
    raise FileNotFoundError(
        "Could not find trained YOLOv8 weights 'best.pt'. "
        "Please provide the path using --weights <path>."
    )


def normalize_labels_dir(labels_dir):
    """
    Ensure all label files use standard 5-token YOLO bounding boxes
    (class_id, x_center, y_center, width, height) instead of mixed polygons,
    preventing Ultralytics from discarding images with mixed-format labels.
    """
    if not os.path.isdir(labels_dir):
        return 0
    
    modified_count = 0
    for root, _, files in os.walk(labels_dir):
        for f in files:
            if not f.endswith('.txt'):
                continue
            fpath = os.path.join(root, f)
            with open(fpath, 'r', encoding='utf-8') as fh:
                lines = fh.readlines()
            
            new_lines = []
            has_poly = False
            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue
                cls_id = parts[0]
                coords = [float(x) for x in parts[1:]]
                if len(coords) == 4:
                    new_lines.append(f"{cls_id} {coords[0]:.6f} {coords[1]:.6f} {coords[2]:.6f} {coords[3]:.6f}\n")
                elif len(coords) > 4:
                    has_poly = True
                    xs = coords[0::2]
                    ys = coords[1::2]
                    x_min, x_max = min(xs), max(xs)
                    y_min, y_max = min(ys), max(ys)
                    w = max(0.001, x_max - x_min)
                    h = max(0.001, y_max - y_min)
                    xc = min(1.0, max(0.0, x_min + w / 2.0))
                    yc = min(1.0, max(0.0, y_min + h / 2.0))
                    new_lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
            
            if has_poly:
                modified_count += 1
                with open(fpath, 'w', encoding='utf-8') as fh:
                    fh.writelines(new_lines)
    
    # Remove old Ultralytics cache so it builds a clean cache
    parent_dir = os.path.dirname(labels_dir)
    cache_path = os.path.join(parent_dir, 'labels.cache')
    if os.path.isfile(cache_path):
        try:
            os.remove(cache_path)
        except Exception:
            pass
            
    return modified_count


def locate_or_prepare_dataset(requested_data=None):
    """
    Search for dataset and prepare a valid Windows-compatible data.yaml.
    If the dataset splits are zipped, extracts them safely.
    """
    project_dir = os.getcwd()
    parent_dir = os.path.dirname(project_dir)

    # 1. If explicit path passed
    if requested_data and os.path.isfile(requested_data):
        return os.path.abspath(requested_data)

    # 2. Check local dataset folder in project
    local_ds = os.path.join(project_dir, 'dataset')
    if os.path.isdir(local_ds) and os.path.isdir(os.path.join(local_ds, 'test', 'images')):
        normalize_labels_dir(os.path.join(local_ds, 'test', 'labels'))
        normalize_labels_dir(os.path.join(local_ds, 'valid', 'labels'))
        yaml_path = create_dataset_yaml(local_ds)
        return yaml_path

    # 3. Check candidate directories and zip archives
    candidate_zips = [
        os.path.join(parent_dir, 'Dataset', 'Fruit Ripeness Detection.v1-fruit-ripeness-detection-dataset.yolov8.zip'),
        os.path.join(parent_dir, 'Dataset Roboflow', 'Fruit Ripeness Detection.v1-fruit-ripeness-detection-dataset.yolov8.zip')
    ]

    for cz in candidate_zips:
        if os.path.isfile(cz):
            print(f"[Dataset] Found dataset archive: {cz}")
            print(f"[Dataset] Extracting evaluation splits to: {local_ds}")
            os.makedirs(local_ds, exist_ok=True)
            win_target = '\\\\?\\' + os.path.abspath(local_ds)
            
            with zipfile.ZipFile(cz, 'r') as z:
                eval_members = [m for m in z.namelist() if m.startswith(('test/', 'valid/')) or m == 'data.yaml']
                for info in eval_members:
                    out_path = os.path.join(win_target, info.replace('/', os.sep))
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)
                    if not info.endswith('/'):
                        with z.open(info) as src, open(out_path, 'wb') as dst:
                            dst.write(src.read())

            normalize_labels_dir(os.path.join(local_ds, 'test', 'labels'))
            normalize_labels_dir(os.path.join(local_ds, 'valid', 'labels'))
            yaml_path = create_dataset_yaml(local_ds)
            return yaml_path

    raise FileNotFoundError(
        "Could not automatically locate the fruit maturity dataset or data.yaml. "
        "Please provide the path using --data <path/to/data.yaml>."
    )


def create_dataset_yaml(dataset_dir):
    """Generate an absolute-path Windows-safe data.yaml configuration."""
    abs_dir = os.path.abspath(dataset_dir).replace('\\', '/')
    cfg = {
        'path': abs_dir,
        'train': 'valid/images',  # Ultralytics requires 'train' key in yaml
        'val': 'valid/images',
        'test': 'test/images',
        'nc': len(ALL_CLASSES),
        'names': ALL_CLASSES
    }
    out_yaml = os.path.join(dataset_dir, 'data.yaml')
    with open(out_yaml, 'w', encoding='utf-8') as f:
        yaml.dump(cfg, f, sort_keys=False)
    return out_yaml


def count_ground_truth_instances(labels_dir, num_classes=32):
    """Count actual ground-truth instances per class in the split."""
    counts = {i: 0 for i in range(num_classes)}
    if not os.path.isdir(labels_dir):
        return counts
    
    for f in os.listdir(labels_dir):
        if not f.endswith('.txt'):
            continue
        p = os.path.join(labels_dir, f)
        with open(p, 'r', encoding='utf-8') as fh:
            for line in fh:
                parts = line.strip().split()
                if parts:
                    try:
                        cid = int(parts[0])
                        if 0 <= cid < num_classes:
                            counts[cid] += 1
                    except ValueError:
                        pass
    return counts


def parse_fruit_and_maturity(class_name):
    """Split class name into (fruit_type, maturity_stage)."""
    for f in FRUIT_TYPES:
        if class_name.startswith(f):
            maturity = class_name[len(f):].strip()
            return f, maturity
    return 'Unknown', 'Unknown'


def plot_fruit_summary(fruit_metrics, save_path):
    """Create grouped bar chart comparing performance across fruit types."""
    fruits = list(fruit_metrics.keys())
    map50 = [fruit_metrics[f]['mAP50'] for f in fruits]
    map50_95 = [fruit_metrics[f]['mAP50-95'] for f in fruits]

    x = np.arange(len(fruits))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=200)
    rects1 = ax.bar(x - width/2, map50, width, label='mAP@50', color='#527A43', edgecolor='#193D2B')
    rects2 = ax.bar(x + width/2, map50_95, width, label='mAP@50-95', color='#E7B955', edgecolor='#8A6546')

    ax.set_ylabel('Score (0.0 – 1.0)', fontsize=12, fontweight='bold')
    ax.set_title('HarvestLens AI - Performance by Fruit Type', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(fruits, fontsize=11, fontweight='bold')
    ax.set_ylim(0, 1.08)
    ax.legend(frameon=True, facecolor='#F8F6ED')
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_maturity_summary(maturity_metrics, save_path):
    """Create grouped bar chart comparing performance across maturity stages."""
    stages = MATURITY_STAGES
    map50 = [maturity_metrics[s]['mAP50'] for s in stages]
    map50_95 = [maturity_metrics[s]['mAP50-95'] for s in stages]
    precision = [maturity_metrics[s]['precision'] for s in stages]
    recall = [maturity_metrics[s]['recall'] for s in stages]

    x = np.arange(len(stages))
    width = 0.2

    fig, ax = plt.subplots(figsize=(9, 5), dpi=200)
    ax.bar(x - 1.5*width, precision, width, label='Precision', color='#35a85a', edgecolor='#193D2B')
    ax.bar(x - 0.5*width, recall, width, label='Recall', color='#4285f4', edgecolor='#193D2B')
    ax.bar(x + 0.5*width, map50, width, label='mAP@50', color='#E7B955', edgecolor='#193D2B')
    ax.bar(x + 1.5*width, map50_95, width, label='mAP@50-95', color='#8A6546', edgecolor='#193D2B')

    ax.set_ylabel('Score (0.0 – 1.0)', fontsize=12, fontweight='bold')
    ax.set_title('HarvestLens AI - Performance by Maturity Stage', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(stages, fontsize=11, fontweight='bold')
    ax.set_ylim(0, 1.12)
    ax.legend(frameon=True, facecolor='#F8F6ED', loc='upper right')
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_per_class_map50(class_data, save_path):
    """Create a ranked horizontal bar chart of mAP@50 for all 32 classes."""
    sorted_classes = sorted(class_data, key=lambda c: c['map50'], reverse=True)
    names = [c['name'] for c in sorted_classes]
    scores = [c['map50'] for c in sorted_classes]
    colors = [STAGE_COLORS.get(c['maturity'], '#527A43') for c in sorted_classes]

    fig, ax = plt.subplots(figsize=(12, 11), dpi=200)
    bars = ax.barh(names, scores, color=colors, edgecolor='#27332A', height=0.68)
    ax.set_xlabel('mAP@50 Score', fontsize=12, fontweight='bold')
    ax.set_title('HarvestLens AI - 32-Class Detection Accuracy Ranking (mAP@50)', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlim(0, 1.08)
    ax.invert_yaxis()  # highest score at top
    ax.grid(axis='x', linestyle='--', alpha=0.5)

    for bar, score in zip(bars, scores):
        ax.text(score + 0.01, bar.get_y() + bar.get_height()/2, f'{score:.3f}',
                va='center', ha='left', fontsize=8.5, fontweight='bold')

    # Legend for maturity stages
    legend_elements = [plt.Rectangle((0, 0), 1, 1, color=col, edgecolor='#27332A', label=stg)
                       for stg, col in STAGE_COLORS.items()]
    ax.legend(handles=legend_elements, title='Maturity Stage', loc='lower right', frameon=True, facecolor='#F8F6ED')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def run_evaluation():
    parser = argparse.ArgumentParser(description="HarvestLens AI - Automated YOLOv8 Evaluation System")
    parser.add_argument('--weights', type=str, default=None, help="Path to best.pt model weights")
    parser.add_argument('--data', type=str, default=None, help="Path to data.yaml or dataset directory")
    parser.add_argument('--split', type=str, default='test', choices=['test', 'val'], help="Dataset split (test or val)")
    parser.add_argument('--imgsz', type=int, default=640, help="Evaluation image size (default: 640)")
    parser.add_argument('--batch', type=int, default=16, help="Batch size for validation (default: 16)")
    parser.add_argument('--conf', type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    parser.add_argument('--iou', type=float, default=0.60, help="NMS IoU threshold (default: 0.60)")
    parser.add_argument('--output', type=str, default='evaluation/reports', help="Output directory for reports")
    parser.add_argument('--device', type=str, default='cpu', help="Compute device ('cpu' or GPU id '0')")
    parser.add_argument('--no-plots', action='store_true', help="Skip generating evaluation plots")
    args = parser.parse_args()

    print("=" * 76)
    print(" HARVESTLENS AI - AUTOMATED MODEL EVALUATION SYSTEM")
    print("=" * 76)

    # 1. Locate weights
    weights_path = find_model_weights(args.weights)
    print(f"[Model] Weights located at: {weights_path}")

    # 2. Locate and verify dataset
    yaml_path = locate_or_prepare_dataset(args.data)
    print(f"[Dataset] Configuration file: {yaml_path}")

    with open(yaml_path, 'r', encoding='utf-8') as f:
        data_cfg = yaml.safe_load(f)

    ds_root = data_cfg.get('path', os.path.dirname(yaml_path))
    split_rel = data_cfg.get(args.split) or ('test/images' if args.split == 'test' else 'valid/images')
    images_dir = os.path.join(ds_root, split_rel)
    labels_dir = images_dir.replace('images', 'labels')

    if not os.path.isdir(images_dir):
        if args.split == 'test':
            print(f"[Warning] Test split not found at {images_dir}. Falling back to validation split.")
            args.split = 'val'
            split_rel = data_cfg.get('val', 'valid/images')
            images_dir = os.path.join(ds_root, split_rel)
            labels_dir = images_dir.replace('images', 'labels')

    img_count = len([f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
    lbl_count = len([f for f in os.listdir(labels_dir) if f.endswith('.txt')]) if os.path.isdir(labels_dir) else 0

    print(f"[Split] Evaluating on '{args.split.upper()}' split:")
    print(f"        Images found:  {img_count}")
    print(f"        Labels found:  {lbl_count}")

    if img_count == 0 or lbl_count == 0:
        raise RuntimeError(f"Split '{args.split}' does not contain valid images and labels.")

    # 3. Ground truth class instances count
    gt_counts = count_ground_truth_instances(labels_dir, len(ALL_CLASSES))

    # 4. Initialize output directory
    output_dir = os.path.abspath(args.output)
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # 5. Load model and run Ultralytics validation
    print(f"\n[Execution] Initializing YOLOv8 inference engine on device: {args.device}...")
    model = YOLO(weights_path)

    t0 = time.time()
    val_results = model.val(
        data=yaml_path,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        plots=True,
        workers=0
    )
    eval_duration = time.time() - t0

    # 6. Extract Global Metrics
    box = val_results.box
    global_p = float(box.mp)
    global_r = float(box.mr)
    global_map50 = float(box.map50)
    global_map50_95 = float(box.map)
    global_f1 = (2 * global_p * global_r / (global_p + global_r)) if (global_p + global_r) > 0 else 0.0

    speed = val_results.speed
    preprocess_ms = float(speed.get('preprocess', 0.0))
    inference_ms = float(speed.get('inference', 0.0))
    postprocess_ms = float(speed.get('postprocess', 0.0))
    total_latency_ms = preprocess_ms + inference_ms + postprocess_ms

    # 7. Extract Per-Class Metrics
    eval_classes = list(box.classes) if hasattr(box, 'classes') and box.classes is not None else list(range(len(ALL_CLASSES)))
    p_arr = box.p if hasattr(box, 'p') and box.p is not None else np.zeros(len(eval_classes))
    r_arr = box.r if hasattr(box, 'r') and box.r is not None else np.zeros(len(eval_classes))
    ap50_arr = box.ap50 if hasattr(box, 'ap50') and box.ap50 is not None else np.zeros(len(eval_classes))
    ap_arr = box.ap if hasattr(box, 'ap') and box.ap is not None else np.zeros(len(eval_classes))

    class_data = []
    for idx, cname in enumerate(ALL_CLASSES):
        fruit, maturity = parse_fruit_and_maturity(cname)
        gt_inst = gt_counts.get(idx, 0)
        
        if idx in eval_classes:
            pos_idx = list(eval_classes).index(idx)
            c_p = float(p_arr[pos_idx])
            c_r = float(r_arr[pos_idx])
            c_ap50 = float(ap50_arr[pos_idx])
            c_ap = float(ap_arr[pos_idx])
            c_f1 = (2 * c_p * c_r / (c_p + c_r)) if (c_p + c_r) > 0 else 0.0
        else:
            c_p, c_r, c_f1, c_ap50, c_ap = 0.0, 0.0, 0.0, 0.0, 0.0

        class_data.append({
            'class_id': idx,
            'name': cname,
            'fruit': fruit,
            'maturity': maturity,
            'gt_instances': gt_inst,
            'precision': round(c_p, 4),
            'recall': round(c_r, 4),
            'f1': round(c_f1, 4),
            'map50': round(c_ap50, 4),
            'map50_95': round(c_ap, 4),
            'status': 'Evaluated' if gt_inst > 0 else 'No ground-truth instances in split'
        })

    # 8. Compute Grouped Metrics
    fruit_groups = {}
    for f in FRUIT_TYPES:
        f_classes = [c for c in class_data if c['fruit'] == f]
        insts = sum(c['gt_instances'] for c in f_classes)
        f_p = np.mean([c['precision'] for c in f_classes]) if f_classes else 0.0
        f_r = np.mean([c['recall'] for c in f_classes]) if f_classes else 0.0
        f_f1 = np.mean([c['f1'] for c in f_classes]) if f_classes else 0.0
        f_50 = np.mean([c['map50'] for c in f_classes]) if f_classes else 0.0
        f_all = np.mean([c['map50_95'] for c in f_classes]) if f_classes else 0.0
        fruit_groups[f] = {
            'instances': insts,
            'precision': round(float(f_p), 4),
            'recall': round(float(f_r), 4),
            'f1': round(float(f_f1), 4),
            'mAP50': round(float(f_50), 4),
            'mAP50-95': round(float(f_all), 4)
        }

    maturity_groups = {}
    for m in MATURITY_STAGES:
        m_classes = [c for c in class_data if c['maturity'] == m]
        insts = sum(c['gt_instances'] for c in m_classes)
        m_p = np.mean([c['precision'] for c in m_classes]) if m_classes else 0.0
        m_r = np.mean([c['recall'] for c in m_classes]) if m_classes else 0.0
        m_f1 = np.mean([c['f1'] for c in m_classes]) if m_classes else 0.0
        m_50 = np.mean([c['map50'] for c in m_classes]) if m_classes else 0.0
        m_all = np.mean([c['map50_95'] for c in m_classes]) if m_classes else 0.0
        maturity_groups[m] = {
            'instances': insts,
            'precision': round(float(m_p), 4),
            'recall': round(float(m_r), 4),
            'f1': round(float(m_f1), 4),
            'mAP50': round(float(m_50), 4),
            'mAP50-95': round(float(m_all), 4)
        }

    # 9. Copy Ultralytics Plots if generated
    if hasattr(val_results, 'save_dir') and os.path.isdir(val_results.save_dir):
        saved_dir = str(val_results.save_dir)
        for item in os.listdir(saved_dir):
            src = os.path.join(saved_dir, item)
            dst = os.path.join(plots_dir, item)
            if os.path.isfile(src) and (src.endswith('.png') or src.endswith('.jpg')):
                shutil.copy2(src, dst)
        print(f"[Artifacts] Ultralytics curves & plots copied to: {plots_dir}")

    # 10. Generate Custom Visualization Plots
    if not args.no_plots:
        plot_fruit_summary(fruit_groups, os.path.join(plots_dir, 'fruit_group_summary.png'))
        plot_maturity_summary(maturity_groups, os.path.join(plots_dir, 'maturity_stage_summary.png'))
        plot_per_class_map50(class_data, os.path.join(plots_dir, 'per_class_map50_chart.png'))
        print(f"[Artifacts] Custom comparative graphs saved to: {plots_dir}")

    # 11. Write CSV Report
    csv_path = os.path.join(output_dir, 'per_class_metrics.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Class_ID', 'Class_Name', 'Fruit', 'Maturity_Stage', 'GT_Instances',
            'Precision', 'Recall', 'F1_Score', 'mAP_50', 'mAP_50_95', 'Status'
        ])
        for c in class_data:
            writer.writerow([
                c['class_id'], c['name'], c['fruit'], c['maturity'], c['gt_instances'],
                f"{c['precision']:.4f}", f"{c['recall']:.4f}", f"{c['f1']:.4f}",
                f"{c['map50']:.4f}", f"{c['map50_95']:.4f}", c['status']
            ])
    print(f"[Artifacts] Per-class CSV metrics saved to: {csv_path}")

    # 12. Write JSON Metrics
    json_path = os.path.join(output_dir, 'metrics.json')
    report_data = {
        'model_name': 'YOLOv8 Fruit Maturity Detector',
        'weights_file': weights_path,
        'evaluation_timestamp': datetime.now().isoformat(),
        'split': args.split,
        'dataset_images': img_count,
        'dataset_ground_truth_instances': sum(c['gt_instances'] for c in class_data),
        'confidence_threshold': args.conf,
        'iou_threshold': args.iou,
        'global_metrics': {
            'precision': round(global_p, 4),
            'recall': round(global_r, 4),
            'f1_score': round(global_f1, 4),
            'map50': round(global_map50, 4),
            'map50_95': round(global_map50_95, 4)
        },
        'latency_ms': {
            'preprocess': round(preprocess_ms, 2),
            'inference': round(inference_ms, 2),
            'postprocess': round(postprocess_ms, 2),
            'total': round(total_latency_ms, 2)
        },
        'fruit_group_summary': fruit_groups,
        'maturity_stage_summary': maturity_groups,
        'per_class_metrics': class_data
    }
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2)
    print(f"[Artifacts] Complete JSON report saved to: {json_path}")

    # 13. Write Text Summary & Print to Terminal
    txt_path = os.path.join(output_dir, 'evaluation_summary.txt')
    
    summary_text = f"""================================================================================
 HARVESTLENS AI - MODEL EVALUATION SUMMARY
================================================================================
Timestamp:               {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Model Weights:           {weights_path}
Evaluation Split:        {args.split.upper()} (Genuine Held-Out Test Set)
Evaluated Images:        {img_count}
Ground Truth Instances:  {sum(c['gt_instances'] for c in class_data)}
Confidence Threshold:    {args.conf}
IoU NMS Threshold:       {args.iou}
Evaluation Time:         {eval_duration:.1f} seconds

--------------------------------------------------------------------------------
 OVERALL PERFORMANCE METRICS
--------------------------------------------------------------------------------
 Precision (P):          {global_p*100:6.2f}%   ({global_p:.4f})
 Recall (R):             {global_r*100:6.2f}%   ({global_r:.4f})
 F1-Score:               {global_f1*100:6.2f}%   ({global_f1:.4f})
 mAP@50:                 {global_map50*100:6.2f}%   ({global_map50:.4f})
 mAP@50-95:              {global_map50_95*100:6.2f}%   ({global_map50_95:.4f})

Inference Latency:       {inference_ms:.1f} ms per image
Total Processing Time:   {total_latency_ms:.1f} ms ({1000/total_latency_ms:.1f} FPS on {args.device.upper()})

--------------------------------------------------------------------------------
 PERFORMANCE BY FRUIT TYPE (8 Varieties)
--------------------------------------------------------------------------------
 Fruit Type    Instances   Precision    Recall      F1-Score    mAP@50    mAP@50-95
--------------------------------------------------------------------------------
"""
    for f in FRUIT_TYPES:
        m = fruit_groups[f]
        summary_text += f" {f:<12} {m['instances']:<10} {m['precision']*100:6.2f}%    {m['recall']*100:6.2f}%    {m['f1']*100:6.2f}%    {m['mAP50']*100:6.2f}%    {m['mAP50-95']*100:6.2f}%\n"

    summary_text += """--------------------------------------------------------------------------------
 PERFORMANCE BY MATURITY STAGE (4 Stages)
--------------------------------------------------------------------------------
 Stage         Instances   Precision    Recall      F1-Score    mAP@50    mAP@50-95
--------------------------------------------------------------------------------
"""
    for s in MATURITY_STAGES:
        m = maturity_groups[s]
        summary_text += f" {s:<12} {m['instances']:<10} {m['precision']*100:6.2f}%    {m['recall']*100:6.2f}%    {m['f1']*100:6.2f}%    {m['mAP50']*100:6.2f}%    {m['mAP50-95']*100:6.2f}%\n"

    summary_text += """--------------------------------------------------------------------------------
 COMPLETE 32-CLASS INVENTORY BREAKDOWN
--------------------------------------------------------------------------------
 ID   Class Name             GT Inst   Precision    Recall      F1-Score    mAP@50    mAP@50-95
--------------------------------------------------------------------------------
"""
    for c in class_data:
        status_flag = " *" if c['gt_instances'] == 0 else ""
        summary_text += (f" {c['class_id']:<3} {c['name']:<22} {c['gt_instances']:<8} "
                         f"{c['precision']*100:6.2f}%    {c['recall']*100:6.2f}%    {c['f1']*100:6.2f}%    "
                         f"{c['map50']*100:6.2f}%    {c['map50_95']*100:6.2f}%{status_flag}\n")

    summary_text += """--------------------------------------------------------------------------------
* Classes marked with (*) have 0 ground-truth annotations in this split.
================================================================================
"""
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(summary_text)

    print("\n" + summary_text)
    print(f"[Done] All evaluation artifacts successfully saved to: {output_dir}")


if __name__ == '__main__':
    run_evaluation()
