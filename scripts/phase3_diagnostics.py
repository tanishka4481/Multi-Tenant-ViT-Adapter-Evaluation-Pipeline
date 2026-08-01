import os
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.ops import box_iou, nms
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from transformers import ViTModel

# --- Local Module Imports ---
from dataset import DynamicDetectionDataset, IsolatedRetailDataset, detection_collate_fn
from train import PureRetailClassificationHead, GridDetectionHead

GRID_SIZE = 14
device = "cuda" if torch.cuda.is_available() else "cpu"


def decode_cell_relative_boxes(pred_box, grid_size=GRID_SIZE):
    """Decodes grid cell predictions [196, 4] into global normalized coordinates [0, 1]."""
    if pred_box.dim() == 3:
        pred_box = pred_box.squeeze(0)

    dx, dy, w, h = pred_box[:, 0], pred_box[:, 1], pred_box[:, 2], pred_box[:, 3]
    dev = pred_box.device
    grid_y, grid_x = torch.meshgrid(
        torch.arange(grid_size, device=dev),
        torch.arange(grid_size, device=dev),
        indexing="ij"
    )
    gx, gy = grid_x.reshape(-1).float(), grid_y.reshape(-1).float()

    xc = (gx + dx) / grid_size
    yc = (gy + dy) / grid_size

    xmin = torch.clamp(xc - (w / 2.0), 0.0, 1.0)
    ymin = torch.clamp(yc - (h / 2.0), 0.0, 1.0)
    xmax = torch.clamp(xc + (w / 2.0), 0.0, 1.0)
    ymax = torch.clamp(yc + (h / 2.0), 0.0, 1.0)

    return torch.stack([ymin, xmin, ymax, xmax], dim=-1)


def convert_yxyx_to_xyxy(boxes_yxyx):
    """Converts [ymin, xmin, ymax, xmax] -> [xmin, ymin, xmax, ymax]."""
    if not isinstance(boxes_yxyx, torch.Tensor):
        boxes_yxyx = torch.tensor(boxes_yxyx, dtype=torch.float32)
    if boxes_yxyx.numel() == 0:
        return boxes_yxyx.reshape(0, 4)
    return boxes_yxyx[:, [1, 0, 3, 2]]


class Phase3DiagnosticEngine:
    def __init__(self, output_dir="outputs/diagnostics_phase3"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.backbone = ViTModel.from_pretrained("google/vit-base-patch16-224").to(device)
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

    @torch.no_grad()
    def evaluate_detection_tenant(self, domain, conf_thresh=0.25, iou_thresh=0.45):
        print(f"\n[PHASE 3] Evaluating Detection Domain: [{domain.upper()}]...")
        domain_output = os.path.join(self.output_dir, domain)
        os.makedirs(domain_output, exist_ok=True)

        dataset = DynamicDetectionDataset(domain, "train")
        loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=detection_collate_fn)

        head = GridDetectionHead(input_dim=768, num_classes=dataset.num_classes).to(device)
        adapter_path = f"outputs/adapters/{domain}_head.pt"
        if os.path.exists(adapter_path):
            head.load_state_dict(torch.load(adapter_path, map_location=device))
            print(f" -> Loaded adapter: {adapter_path}")
        head.eval()

        metric = MeanAveragePrecision(iou_type="bbox")
        all_ious = []
        y_true, y_pred = [], []
        bg_idx = dataset.num_classes
        full_class_names = dataset.class_names + ["background"]

        for batch in loader:
            pixels = batch["pixel_values"].to(device)
            gt_boxes_raw = batch["boxes"][0]
            gt_classes_raw = batch["classes"][0]

            patch_tokens = self.backbone(pixels).last_hidden_state[:, 1:, :]
            pred_obj, pred_box, pred_cls = head(patch_tokens)

            obj_scores = torch.sigmoid(pred_obj[0].squeeze(-1))
            cls_probs = torch.softmax(pred_cls[0], dim=-1)
            decoded_boxes = decode_cell_relative_boxes(pred_box[0], GRID_SIZE)

            keep_mask = obj_scores >= conf_thresh
            keep_indices = torch.where(keep_mask)[0]

            if len(keep_indices) > 0:
                p_boxes_yxyx = decoded_boxes[keep_indices]
                p_scores = obj_scores[keep_indices]
                p_classes = cls_probs[keep_indices].argmax(dim=-1)

                p_boxes_xyxy = convert_yxyx_to_xyxy(p_boxes_yxyx)
                nms_keep = nms(p_boxes_xyxy, p_scores, iou_thresh)

                p_boxes_yxyx = p_boxes_yxyx[nms_keep]
                p_scores = p_scores[nms_keep]
                p_classes = p_classes[nms_keep]
            else:
                p_boxes_yxyx = torch.zeros((0, 4), device=device)
                p_scores = torch.zeros((0,), device=device)
                p_classes = torch.zeros((0,), dtype=torch.long, device=device)

            # Convert to absolute pixel scales (224x224) for mAP calculation
            p_boxes_map = convert_yxyx_to_xyxy(p_boxes_yxyx) * 224.0
            gt_boxes_map = convert_yxyx_to_xyxy(torch.tensor(gt_boxes_raw, device=device)) * 224.0
            gt_labels_map = torch.tensor(gt_classes_raw, dtype=torch.long, device=device)

            metric.update(
                [{"boxes": p_boxes_map.cpu(), "scores": p_scores.cpu(), "labels": p_classes.cpu()}],
                [{"boxes": gt_boxes_map.cpu(), "labels": gt_labels_map.cpu()}]
            )

            # Track IoUs and Confusion Matrices
            if len(gt_boxes_map) > 0 and len(p_boxes_map) > 0:
                ious = box_iou(p_boxes_map, gt_boxes_map)
                max_ious, matched_gt_idx = ious.max(dim=1)
                all_ious.extend(max_ious.cpu().tolist())

                matched_pred_mask = torch.zeros(len(p_boxes_map), dtype=torch.bool)
                for g_i, g_cls in enumerate(gt_labels_map.cpu().tolist()):
                    if ious.size(0) > 0:
                        max_iou_val, p_i = ious[:, g_i].max(dim=0)
                        if max_iou_val >= 0.5:
                            y_true.append(g_cls)
                            y_pred.append(p_classes[p_i].item())
                            matched_pred_mask[p_i] = True
                        else:
                            y_true.append(g_cls)
                            y_pred.append(bg_idx)

                for p_i, is_matched in enumerate(matched_pred_mask):
                    if not is_matched:
                        y_true.append(bg_idx)
                        y_pred.append(p_classes[p_i].item())
            elif len(gt_boxes_map) > 0 and len(p_boxes_map) == 0:
                for g_cls in gt_labels_map.cpu().tolist():
                    y_true.append(g_cls)
                    y_pred.append(bg_idx)

        # Compute mAP Metrics
        map_results = metric.compute()
        map_summary = {
            "mAP_50": round(map_results["map_50"].item(), 4),
            "mAP_50_95": round(map_results["map"].item(), 4),
            "mar_100": round(map_results["mar_100"].item(), 4)
        }

        with open(f"{domain_output}/map_metrics.json", "w") as f:
            json.dump(map_summary, f, indent=4)

        # Plot IoU Distribution Histogram
        if len(all_ious) > 0:
            plt.figure(figsize=(6, 4))
            plt.hist(all_ious, bins=20, color="teal", alpha=0.7, edgecolor="black")
            plt.xlabel("Intersection over Union (IoU)")
            plt.ylabel("Prediction Count")
            plt.title(f"Bounding Box IoU Distribution — {domain.upper()}")
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.tight_layout()
            plt.savefig(f"{domain_output}/iou_distribution.png")
            plt.close()

        # Plot Confusion Matrix
        if len(y_true) > 0:
            cm = confusion_matrix(y_true, y_pred, labels=list(range(len(full_class_names))))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=full_class_names)
            fig, ax = plt.subplots(figsize=(7, 7))
            disp.plot(ax=ax, cmap="Purples", colorbar=False)
            plt.title(f"Phase 3 Confusion Matrix w/ Background — {domain.upper()}")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f"{domain_output}/confusion_matrix.png")
            plt.close()

        print(f"✅ [{domain.upper()}] Evaluation Complete. mAP@0.50: {map_summary['mAP_50']}")
        return map_summary

    @torch.no_grad()
    def evaluate_retail_tenant(self):
        print(f"\n[PHASE 3] Evaluating Classification Domain: [RETAIL]...")
        domain_output = os.path.join(self.output_dir, "retail")
        os.makedirs(domain_output, exist_ok=True)

        dataset = IsolatedRetailDataset("train")
        loader = DataLoader(dataset, batch_size=16, shuffle=False)

        head = PureRetailClassificationHead(input_dim=768, num_classes=len(dataset.class_to_idx)).to(device)
        adapter_path = "outputs/adapters/retail_head.pt"
        if os.path.exists(adapter_path):
            head.load_state_dict(torch.load(adapter_path, map_location=device))
            print(f" -> Loaded adapter: {adapter_path}")
        head.eval()

        all_preds, all_labels = [], []
        for batch in loader:
            pixels = batch["pixel_values"].to(device)
            labels = batch["class_label"].to(device)

            cls_token = self.backbone(pixels).last_hidden_state[:, 0, :]
            outputs = head(cls_token)

            preds = outputs.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        inv_map = {v: k for k, v in dataset.class_to_idx.items()}
        target_names = [inv_map[i] for i in range(len(inv_map))]

        report = classification_report(all_labels, all_preds, target_names=target_names, output_dict=True)
        acc = round(report["accuracy"], 4)

        with open(f"{domain_output}/classification_report.json", "w") as f:
            json.dump(report, f, indent=4)

        cm = confusion_matrix(all_labels, all_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
        fig, ax = plt.subplots(figsize=(6, 6))
        disp.plot(ax=ax, cmap="Blues", colorbar=False)
        plt.title("Retail Classification Confusion Matrix")
        plt.tight_layout()
        plt.savefig(f"{domain_output}/confusion_matrix.png")
        plt.close()

        print(f"✅ [RETAIL] Evaluation Complete. Categorical Accuracy: {acc}")
        return {"accuracy": acc}


def run_phase3_diagnostics():
    engine = Phase3DiagnosticEngine()
    summary = {
        "warehouse": engine.evaluate_detection_tenant("warehouse"),
        "server_room": engine.evaluate_detection_tenant("server_room"),
        "retail": engine.evaluate_retail_tenant()
    }

    with open("outputs/diagnostics_phase3/global_eval_summary.json", "w") as f:
        json.dump(summary, f, indent=4)

    print("\n==========================================")
    print("      PHASE 3 GLOBAL EVALUATION SUMMARY   ")
    print("==========================================")
    print(json.dumps(summary, indent=4))
    print("==========================================")


if __name__ == "__main__":
    run_phase3_diagnostics()
