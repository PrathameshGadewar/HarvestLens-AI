# HarvestLens AI — Model Evaluation Framework

This directory contains the automated evaluation framework for the HarvestLens AI trained YOLOv8 fruit maturity detection model (`best.pt`).

---

## 1. Overview

The HarvestLens AI detector recognizes **32 distinct fruit-maturity classes** across **8 fruit varieties** and **4 ripeness stages**:

| Fruit Varieties | Ripeness / Maturity Stages |
| :--- | :--- |
| **Apple**, **Banana**, **Grape**, **Mango**, **Melon**, **Orange**, **Peach**, **Pear** | **Unripe**, **Ripe**, **Overripe**, **Rotten** |

### Complete 32-Class Inventory:
1. `Apple Overripe`
2. `Apple Ripe`
3. `Apple Rotten`
4. `Apple Unripe`
5. `Banana Overripe`
6. `Banana Ripe`
7. `Banana Rotten`
8. `Banana Unripe`
9. `Grape Overripe`
10. `Grape Ripe`
11. `Grape Rotten`
12. `Grape Unripe`
13. `Mango Overripe`
14. `Mango Ripe`
15. `Mango Rotten`
16. `Mango Unripe`
17. `Melon Overripe`
18. `Melon Ripe`
19. `Melon Rotten`
20. `Melon Unripe`
21. `Orange Overripe`
22. `Orange Ripe`
23. `Orange Rotten`
24. `Orange Unripe`
25. `Peach Overripe`
26. `Peach Ripe`
27. `Peach Rotten`
28. `Peach Unripe`
29. `Pear Overripe`
30. `Pear Ripe`
31. `Pear Rotten`
32. `Pear Unripe`

---

## 2. Evaluation Metrics Explained

### Precision ($P$)
$$\text{Precision} = \frac{\text{True Positives (TP)}}{\text{True Positives (TP)} + \text{False Positives (FP)}}$$
- **Meaning in HarvestLens**: Of all bounding boxes where the model predicted a fruit ripeness stage, what fraction were genuinely that fruit and stage?
- **High Precision** ensures farmers and buyers do not receive false alerts (e.g. an unripe mango misidentified as ripe).

### Recall ($R$)
$$\text{Recall} = \frac{\text{True Positives (TP)}}{\text{True Positives (TP)} + \text{False Negatives (FN)}}$$
- **Meaning in HarvestLens**: Of all actual ground-truth fruits present on the plant or in the crate, what fraction did the YOLOv8 model successfully detect?
- **High Recall** ensures that damaged or ripe fruits are not overlooked during orchard passes.

### $F_1$-Score ($F_1$)
$$F_1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
- The harmonic mean balancing precision and recall. It reflects overall detection reliability when confidence thresholds vary.

### Intersection over Union ($\text{IoU}$)
$$\text{IoU} = \frac{\text{Area of Overlap}}{\text{Area of Union}}$$
- Measures bounding box localization accuracy against the ground truth. A detection is counted as a True Positive if its $\text{IoU} \ge \text{Threshold}$ (default $0.50$).

### mAP@50 (Mean Average Precision at IoU = 0.50)
- The area under the Precision-Recall curve evaluated at an $\text{IoU}$ threshold of $0.50$, averaged across all classes.
- Standard PASCAL VOC-style metric representing detection and classification accuracy under standard localization tolerance.

### mAP@50–95 (COCO Primary Challenge Metric)
- Mean Average Precision averaged across 10 IoU thresholds from $0.50$ to $0.95$ in increments of $0.05$:
  $$\text{mAP@[0.50:0.95]} = \frac{1}{10} \sum_{k=0}^{9} \text{AP@}(0.50 + 0.05k)$$
- Highly penalizes loose or imprecise bounding boxes, demanding both accurate classification and tight bounding-box boundaries.

### Confusion Matrix
- Maps true classes (rows) against predicted classes (columns).
- Distinguishes **Ripeness Confusion** (e.g., *Mango Ripe* vs *Mango Overripe*) from **Fruit Confusion** (e.g., *Apple* vs *Peach*).

---

## 3. How to Run the Evaluation

From the project root directory, run:

```bash
# Evaluate on the genuine held-out test split (default)
python evaluate_model.py

# Evaluate on the validation split
python evaluate_model.py --split val

# Specify custom weights and output directory
python evaluate_model.py --weights models/best.pt --split test --output evaluation/reports

# Evaluate with custom confidence and IoU thresholds
python evaluate_model.py --conf 0.25 --iou 0.60
```

### Command Line Options

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--weights` | string | `best.pt` | Path to trained model weights |
| `--data` | string | Auto-detected | Path to `data.yaml` or dataset directory |
| `--split` | string | `test` | Dataset split to evaluate: `test` or `val` |
| `--batch` | int | `16` | Batch size for inference |
| `--imgsz` | int | `640` | Input image resolution |
| `--conf` | float | `0.25` | Confidence threshold for detection |
| `--iou` | float | `0.60` | NMS IoU threshold |
| `--output` | string | `evaluation/reports` | Output directory for logs, CSVs, and plots |
| `--device` | string | Auto | Device to use (`cpu`, `0`, `cuda`) |

---

## 4. Output Artifacts

Running `evaluate_model.py` generates the following files in `evaluation/reports/`:

1. **`evaluation_summary.txt`**: Human-readable terminal and text summary with global scores and per-class tables.
2. **`per_class_metrics.csv`**: Comprehensive CSV file with metrics for all 32 classes.
3. **`metrics.json`**: Machine-readable JSON structured data containing all metrics, latency figures, and class-level scores.
4. **`confusion_matrix.png`**: High-resolution heatmap showing detection matches and cross-class confusions.
5. **`precision_recall_curve.png`**: PR curve showing precision trade-offs across confidence values.
6. **`fruit_group_summary.png`**: Grouped performance by fruit type (Apple, Banana, Mango, etc.).
7. **`ripeness_stage_summary.png`**: Grouped performance by maturity stage (Unripe, Ripe, Overripe, Rotten).
