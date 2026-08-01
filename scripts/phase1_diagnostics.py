import os
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.ops import box_iou, nms
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from transformers import ViTModel

device = "cuda" if torch.cuda.is_available() else "cpu"
GRID_SIZE = 14

class FullViTGridDetector(nn.Module):
    """Combines frozen ViT backbone and loaded head weights for evaluation."""
    def __init__(self, domain, num_classes):
        super().__init__()
        self.backbone = ViTModel.from_pretrained("google/vit-base-patch16-224")
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()
        
        self.head = GridDetectionHead(input_dim=768, num_classes=num_classes)
        head_path = f"outputs/adapters/{domain}_head.pt"
        if os.path.exists(head_path):
            self.head.load_state_dict(torch.load(head_path, map_location=device))
            print(f"✅ Successfully loaded adapter head weights from {head_path}")
        else:
            print(f"⚠️ Warning: Adapter file {head_path} not found! Running with uninitialized head.")
        self.head.eval()

    @torch.no_grad()
    def forward(self, pixel_values):
        patch_tokens = self.backbone(pixel_values).last_hidden_state[:, 1:, :] # [B, 196, 768]
        pred_obj, pred_box, pred_cls = self.head(patch_tokens)
        return pred_obj, pred_box, pred_cls


class Phase1DiagnosticCollector:
    def __init__(self, model, dataloader, device, num_classes, class_names, domain_name, grid_size=GRID_SIZE):
        self.model = model.to(device)
        self.dataloader = dataloader
        self.device = device
        self.num_classes = num_classes
        self.class_names = class_names
        self.domain_name = domain_name
        self.grid_size = grid_size
        self.output_dir = f"outputs/diagnostics_phase1/{domain_name}"
        os.makedirs(self.output_dir, exist_ok=True)

    @torch.no_grad()
    def collect_raw_and_decoded(self, num_samples=50):
        self.model.eval()
        audit_records = []
        sample_count = 0
        
        for batch in self.dataloader:
            if sample_count >= num_samples:
                break
            
            pixels = batch["pixel_values"].to(self.device)
            raw_obj, raw_box, raw_cls = self.model(pixels)
            
            for b in range(pixels.size(0)):
                obj_scores = torch.sigmoid(raw_obj[b]).squeeze(-1).cpu().numpy()
                boxes_raw = raw_box[b].cpu().numpy()
                cls_probs = torch.softmax(raw_cls[b], dim=-1).cpu().numpy()
                
                decoded_global = boxes_raw.copy()
                decoded_cell_relative = np.zeros_like(boxes_raw)
                
                for cell_idx in range(self.grid_size * self.grid_size):
                    row = cell_idx // self.grid_size
                    col = cell_idx % self.grid_size
                    dx, dy, bw, bh = boxes_raw[cell_idx]
                    
                    xc = (col + dx) / float(self.grid_size)
                    yc = (row + dy) / float(self.grid_size)
                    
                    decoded_cell_relative[cell_idx] = [
                        max(0.0, yc - bh / 2.0),
                        max(0.0, xc - bw / 2.0),
                        min(1.0, yc + bh / 2.0),
                        min(1.0, xc + bw / 2.0)
                    ]

                target_info = {
                    "boxes": batch["boxes"][b],
                    "classes": batch["classes"][b]
                }

                audit_records.append({
                    "image_id": sample_count,
                    "target": target_info,
                    "raw_obj_max": float(obj_scores.max()),
                    "raw_obj_mean": float(obj_scores.mean()),
                    "boxes_global": decoded_global,
                    "boxes_cell_relative": decoded_cell_relative,
                    "obj_scores": obj_scores,
                    "cls_probs": cls_probs
                })
                sample_count += 1

        audit_summary = [{
            "image_id": r["image_id"],
            "raw_obj_max": r["raw_obj_max"],
            "raw_obj_mean": r["raw_obj_mean"],
            "gt_count": len(r["target"]["boxes"])
        } for r in audit_records]
        
        with open(f"{self.output_dir}/decode_audit_summary.json", "w") as f:
            json.dump(audit_summary, f, indent=2)

        print(f"[{self.domain_name.upper()}] ✅ Collected {len(audit_records)} audit samples.")
        return audit_records


def convert_to_xyxy(boxes_yxyx):
    """Converts [ymin, xmin, ymax, xmax] -> [xmin, ymin, xmax, ymax] for torchvision ops."""
    if isinstance(boxes_yxyx, np.ndarray):
        boxes_yxyx = torch.tensor(boxes_yxyx, dtype=torch.float32)
    elif not isinstance(boxes_yxyx, torch.Tensor):
        boxes_yxyx = torch.tensor(boxes_yxyx, dtype=torch.float32)
        
    if boxes_yxyx.numel() == 0:
        return boxes_yxyx.reshape(0, 4)
    
    return boxes_yxyx[:, [1, 0, 3, 2]]


def decode_predictions_phase1_patched(obj_scores, raw_boxes_yxyx, cls_probs, conf_thresh, iou_thresh):
    keep_indices = np.where(obj_scores >= conf_thresh)[0]
    if len(keep_indices) == 0:
        return torch.empty((0, 4)), torch.empty((0,)), torch.empty((0,), dtype=torch.long)

    filtered_scores = torch.tensor(obj_scores[keep_indices], dtype=torch.float32)
    filtered_cls = torch.tensor(cls_probs[keep_indices].argmax(axis=-1), dtype=torch.long)
    
    boxes_yxyx = torch.tensor(raw_boxes_yxyx[keep_indices], dtype=torch.float32)
    boxes_xyxy = convert_to_xyxy(boxes_yxyx)
    
    nms_keep = nms(boxes_xyxy, filtered_scores, iou_thresh)
    return boxes_yxyx[nms_keep], filtered_scores[nms_keep], filtered_cls[nms_keep]


def run_sweeps_phase1_patched(audit_records, domain_name, conf_thresholds=[0.10, 0.25, 0.50], nms_thresholds=[0.30, 0.45, 0.60]):
    results = []
    output_dir = f"outputs/diagnostics_phase1/{domain_name}"
    os.makedirs(output_dir, exist_ok=True)

    for conf in conf_thresholds:
        for iou_t in nms_thresholds:
            total_preds = 0
            total_matched = 0
            total_gt = 0

            for record in audit_records:
                gt_boxes_yxyx = torch.tensor(record["target"]["boxes"], dtype=torch.float32)
                total_gt += len(gt_boxes_yxyx)

                pred_boxes_yxyx, pred_scores, _ = decode_predictions_phase1_patched(
                    record["obj_scores"], record["boxes_global"], record["cls_probs"],
                    conf_thresh=conf, iou_thresh=iou_t
                )
                
                total_preds += len(pred_boxes_yxyx)
                
                if len(pred_boxes_yxyx) > 0 and len(gt_boxes_yxyx) > 0:
                    pred_boxes_xyxy = convert_to_xyxy(pred_boxes_yxyx)
                    gt_boxes_xyxy = convert_to_xyxy(gt_boxes_yxyx)
                    
                    ious = box_iou(pred_boxes_xyxy, gt_boxes_xyxy)
                    matched_gt = (ious.max(dim=0).values >= 0.5).sum().item()
                    total_matched += matched_gt

            avg_preds_per_img = total_preds / max(1, len(audit_records))
            recall = total_matched / max(1, total_gt)
            precision = total_matched / max(1, total_preds)

            results.append({
                "conf_thresh": conf,
                "nms_iou_thresh": iou_t,
                "total_predictions": total_preds,
                "avg_preds_per_image": round(avg_preds_per_img, 2),
                "matched_gt_count": total_matched,
                "precision_proxy": round(precision, 4),
                "recall_proxy": round(recall, 4)
            })

    df = pd.DataFrame(results)
    df.to_csv(f"{output_dir}/threshold_nms_sweep_corrected.csv", index=False)
    print(f"[{domain_name.upper()}] ✅ Corrected Phase 1 Sweep Saved:\n", df.to_string())
    return df


def decode_predictions(obj_scores, raw_boxes, cls_probs, conf_thresh, iou_thresh, is_cell_relative=False, grid_size=14):
    """Decodes raw heads and applies Non-Maximum Suppression (NMS)."""
    keep_indices = np.where(obj_scores >= conf_thresh)[0]
    if len(keep_indices) == 0:
        return torch.empty((0, 4)), torch.empty((0,)), torch.empty((0,), dtype=torch.long)

    filtered_scores = torch.tensor(obj_scores[keep_indices])
    filtered_cls = torch.tensor(cls_probs[keep_indices].argmax(axis=-1))
    
    boxes_list = []
    for idx in keep_indices:
        b = raw_boxes[idx]
        if is_cell_relative:
            row, col = idx // grid_size, idx % grid_size
            xc = (col + b[0]) / float(grid_size)
            yc = (row + b[1]) / float(grid_size)
            w, h = b[2], b[3]
            boxes_list.append([yc - h/2.0, xc - w/2.0, yc + h/2.0, xc + w/2.0])
        else:
            boxes_list.append(b)
            
    boxes_tensor = torch.tensor(np.array(boxes_list), dtype=torch.float32)
    nms_keep = nms(boxes_tensor, filtered_scores, iou_thresh)
    return boxes_tensor[nms_keep], filtered_scores[nms_keep], filtered_cls[nms_keep]


def compute_detection_confusion_matrix(audit_records, class_names, domain_name, conf_thresh=0.25, iou_thresh=0.45):
    output_dir = f"outputs/diagnostics_phase1/{domain_name}"
    y_true, y_pred = [], []
    bg_idx = len(class_names)
    full_labels = class_names + ["background"]

    for record in audit_records:
        gt_boxes = torch.tensor(record["target"]["boxes"], dtype=torch.float32)
        gt_labels = record["target"]["classes"]

        pred_boxes, pred_scores, pred_classes = decode_predictions(
            record["obj_scores"], record["boxes_global"], record["cls_probs"],
            conf_thresh=conf_thresh, iou_thresh=iou_thresh
        )

        if len(gt_boxes) == 0 and len(pred_boxes) > 0:
            for p_cls in pred_classes:
                y_true.append(bg_idx)
                y_pred.append(p_cls.item())
        elif len(gt_boxes) > 0 and len(pred_boxes) == 0:
            for g_cls in gt_labels:
                y_true.append(g_cls)
                y_pred.append(bg_idx)
        elif len(gt_boxes) > 0 and len(pred_boxes) > 0:
            ious = box_iou(gt_boxes, pred_boxes)
            matched_pred_mask = torch.zeros(len(pred_boxes), dtype=torch.bool)

            for g_i, g_cls in enumerate(gt_labels):
                max_iou, p_i = ious[g_i].max(dim=0)
                if max_iou >= 0.5:
                    y_true.append(g_cls)
                    y_pred.append(pred_classes[p_i].item())
                    matched_pred_mask[p_i] = True
                else:
                    y_true.append(g_cls)
                    y_pred.append(bg_idx)

            for p_i, is_matched in enumerate(matched_pred_mask):
                if not is_matched:
                    y_true.append(bg_idx)
                    y_pred.append(pred_classes[p_i].item())

    if len(y_true) > 0:
        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(full_labels))))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=full_labels)
        fig, ax = plt.subplots(figsize=(7, 7))
        disp.plot(ax=ax, cmap="Blues", colorbar=False)
        plt.title(f"Detection Confusion Matrix w/ Background — {domain_name.upper()}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/confusion_matrix.png")
        plt.close()


def plot_pr_curves_and_histograms(audit_records, class_names, domain_name):
    output_dir = f"outputs/diagnostics_phase1/{domain_name}"
    tp_scores, fp_scores = [], []

    for record in audit_records:
        gt_boxes = torch.tensor(record["target"]["boxes"], dtype=torch.float32)
        pred_boxes, pred_scores, _ = decode_predictions(
            record["obj_scores"], record["boxes_global"], record["cls_probs"],
            conf_thresh=0.05, iou_thresh=0.45
        )

        if len(pred_boxes) > 0:
            if len(gt_boxes) == 0:
                fp_scores.extend(pred_scores.tolist())
            else:
                ious = box_iou(pred_boxes, gt_boxes)
                max_ious = ious.max(dim=1).values
                for score, iou_val in zip(pred_scores, max_ious):
                    if iou_val >= 0.5:
                        tp_scores.append(score.item())
                    else:
                        fp_scores.append(score.item())

    plt.figure(figsize=(7, 4))
    plt.hist(tp_scores, bins=20, alpha=0.6, label="True Positives (IoU >= 0.5)", color="green")
    plt.hist(fp_scores, bins=20, alpha=0.6, label="False Positives (IoU < 0.5)", color="red")
    plt.xlabel("Confidence Score")
    plt.ylabel("Count")
    plt.title(f"Objectness Confidence Separability — {domain_name.upper()}")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/score_histogram.png")
    plt.close()


def export_categorized_overlays(audit_records, class_names, domain_name, thresholds=[0.10, 0.25, 0.50]):
    for thresh in thresholds:
        base_dir = f"outputs/diagnostics_phase1/{domain_name}/overlays_thresh_{thresh}"
        os.makedirs(f"{base_dir}/good", exist_ok=True)
        os.makedirs(f"{base_dir}/false_positives", exist_ok=True)
        os.makedirs(f"{base_dir}/false_negatives", exist_ok=True)
        os.makedirs(f"{base_dir}/duplicates", exist_ok=True)

        for rec in audit_records[:15]:
            gt_boxes = rec["target"]["boxes"]
            pred_boxes, pred_scores, pred_cls = decode_predictions(
                rec["obj_scores"], rec["boxes_global"], rec["cls_probs"],
                conf_thresh=thresh, iou_thresh=0.45
            )

            fig, ax = plt.subplots(figsize=(5, 5))
            ax.set_title(f"Img {rec['image_id']} | Thresh {thresh}")
            ax.set_xlim(0, 224)
            ax.set_ylim(224, 0)
            
            for box in gt_boxes:
                ymin, xmin, ymax, xmax = box
                rect = plt.Rectangle((xmin * 224, ymin * 224), (xmax - xmin) * 224, (ymax - ymin) * 224, 
                                     fill=False, edgecolor='green', linewidth=2)
                ax.add_patch(rect)

            for p_box, score, cls_i in zip(pred_boxes, pred_scores, pred_cls):
                ymin, xmin, ymax, xmax = p_box.numpy()
                rect = plt.Rectangle((xmin * 224, ymin * 224), (xmax - xmin) * 224, (ymax - ymin) * 224, 
                                     fill=False, edgecolor='red', linewidth=1.5, linestyle="--")
                ax.add_patch(rect)
                ax.text(xmin * 224, ymin * 224, f"{class_names[cls_i]}:{score:.2f}", 
                        color='white', backgroundcolor='red', fontsize=8)

            plt.axis("off")
            
            if len(gt_boxes) > 0 and len(pred_boxes) > 0:
                plt.savefig(f"{base_dir}/good/img_{rec['image_id']}.png", bbox_inches='tight')
            elif len(pred_boxes) > len(gt_boxes):
                plt.savefig(f"{base_dir}/duplicates/img_{rec['image_id']}.png", bbox_inches='tight')
            elif len(pred_boxes) == 0:
                plt.savefig(f"{base_dir}/false_negatives/img_{rec['image_id']}.png", bbox_inches='tight')
            else:
                plt.savefig(f"{base_dir}/false_positives/img_{rec['image_id']}.png", bbox_inches='tight')
                
            plt.close()


def run_phase1_pipeline():
    domains = ["server_room", "warehouse"]

    for domain in domains:
        print(f"\n==========================================")
        print(f"  RUNNING PHASE 1 DIAGNOSTICS: {domain.upper()}")
        print(f"==========================================")

        dataset = DynamicDetectionDataset(domain, "train")
        loader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=detection_collate_fn)

        num_classes = dataset.num_classes
        class_names = dataset.class_names

        print(f" -> Dynamically resolved metadata for {domain}:")
        print(f"    Classes ({num_classes}): {class_names}")

        model = FullViTGridDetector(domain=domain, num_classes=num_classes).to(device)

        collector = Phase1DiagnosticCollector(
            model=model,
            dataloader=loader,
            device=device,
            num_classes=num_classes,
            class_names=class_names,
            domain_name=domain
        )

        records = collector.collect_raw_and_decoded(num_samples=40)
        run_sweeps_phase1_patched(records, domain)
        compute_detection_confusion_matrix(records, class_names, domain)
        plot_pr_curves_and_histograms(records, class_names, domain)
        export_categorized_overlays(records, class_names, domain)

    print("\n==========================================")
    print("  ✅ PHASE 1 COMPLETE FOR ALL DOMAINS")
    print("  Check 'outputs/diagnostics_phase1/'")
    print("==========================================")


if __name__ == "__main__":
    run_phase1_pipeline()
