import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torchvision.ops import box_iou, nms
from transformers import ViTModel

GRID_SIZE = 14
device = "cuda" if torch.cuda.is_available() else "cpu"

class FullViTGridDetector(nn.Module):
    """Combines frozen ViT backbone and domain adapter head for evaluation."""
    def __init__(self, domain, num_classes):
        super().__init__()
        self.backbone = ViTModel.from_pretrained("google/vit-base-patch16-224")
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

        self.head = GridDetectionHead(input_dim=768, num_classes=num_classes)
        adapter_path = f"outputs/adapters/{domain}_head.pt"
        if os.path.exists(adapter_path):
            self.head.load_state_dict(torch.load(adapter_path, map_location=device))
            print(f" -> Loaded trained adapter: {adapter_path}")
        else:
            print(f" ⚠️ Warning: Adapter weights not found at {adapter_path}. Using initial weights.")

    def forward(self, x):
        with torch.no_grad():
            patch_tokens = self.backbone(x).last_hidden_state[:, 1:, :]
        return self.head(patch_tokens)


def convert_yxyx_to_xyxy(boxes_yxyx):
    """Converts [ymin, xmin, ymax, xmax] -> [xmin, ymin, xmax, ymax] for torchvision ops."""
    if not isinstance(boxes_yxyx, torch.Tensor):
        boxes_yxyx = torch.tensor(boxes_yxyx, dtype=torch.float32)
    if boxes_yxyx.numel() == 0:
        return boxes_yxyx.reshape(0, 4)
    return boxes_yxyx[:, [1, 0, 3, 2]]


def decode_cell_relative_boxes(pred_box, grid_size=GRID_SIZE):
    """
    Decodes grid cell predictions [196, 4] into global normalized coordinates [0, 1].
    Expects pred_box outputs where offsets/sizes are already bounded in range [0, 1].
    """
    if pred_box.dim() == 3:
        pred_box = pred_box.squeeze(0)

    dx = pred_box[:, 0]
    dy = pred_box[:, 1]
    w  = pred_box[:, 2]
    h  = pred_box[:, 3]
    
    dev = pred_box.device
    grid_y, grid_x = torch.meshgrid(
        torch.arange(grid_size, device=dev),
        torch.arange(grid_size, device=dev),
        indexing="ij"
    )
    gx = grid_x.reshape(-1).float()
    gy = grid_y.reshape(-1).float()
    
    xc = (gx + dx) / grid_size
    yc = (gy + dy) / grid_size
    
    xmin = torch.clamp(xc - (w / 2.0), 0.0, 1.0)
    ymin = torch.clamp(yc - (h / 2.0), 0.0, 1.0)
    xmax = torch.clamp(xc + (w / 2.0), 0.0, 1.0)
    ymax = torch.clamp(yc + (h / 2.0), 0.0, 1.0)
    
    return torch.stack([ymin, xmin, ymax, xmax], dim=-1)


class Phase2DiagnosticCollector:
    def __init__(self, model, dataloader, device, num_classes, class_names, domain_name, grid_size=GRID_SIZE):
        self.model = model.to(device)
        self.dataloader = dataloader
        self.device = device
        self.num_classes = num_classes
        self.class_names = class_names
        self.domain_name = domain_name
        self.grid_size = grid_size
        self.output_dir = f"outputs/diagnostics_phase2/{domain_name}"
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
                cls_probs = torch.softmax(raw_cls[b], dim=-1).cpu().numpy()
                decoded_boxes = decode_cell_relative_boxes(raw_box[b], self.grid_size).cpu().numpy()

                target_info = {
                    "boxes": batch["boxes"][b],
                    "classes": batch["classes"][b]
                }

                audit_records.append({
                    "image_id": sample_count,
                    "target": target_info,
                    "raw_obj_max": float(obj_scores.max()),
                    "raw_obj_mean": float(obj_scores.mean()),
                    "boxes_decoded": decoded_boxes, # Output in [ymin, xmin, ymax, xmax]
                    "obj_scores": obj_scores,
                    "cls_probs": cls_probs
                })
                sample_count += 1

        print(f"[{self.domain_name.upper()}] ✅ Collected {len(audit_records)} audit samples with cell-relative decoder.")
        return audit_records


def decode_predictions_phase2(obj_scores, decoded_boxes_yxyx, cls_probs, conf_thresh, iou_thresh):
    keep_indices = np.where(obj_scores >= conf_thresh)[0]
    if len(keep_indices) == 0:
        return torch.empty((0, 4)), torch.empty((0,)), torch.empty((0,), dtype=torch.long)

    filtered_scores = torch.tensor(obj_scores[keep_indices], dtype=torch.float32)
    filtered_cls = torch.tensor(cls_probs[keep_indices].argmax(axis=-1), dtype=torch.long)
    boxes_yxyx_tensor = torch.tensor(decoded_boxes_yxyx[keep_indices], dtype=torch.float32)
    
    boxes_xyxy_tensor = convert_yxyx_to_xyxy(boxes_yxyx_tensor)
    nms_keep = nms(boxes_xyxy_tensor, filtered_scores, iou_thresh)
    return boxes_yxyx_tensor[nms_keep], filtered_scores[nms_keep], filtered_cls[nms_keep]


def run_sweeps_phase2(audit_records, domain_name, conf_thresholds=[0.10, 0.25, 0.50], nms_thresholds=[0.30, 0.45, 0.60]):
    results = []
    output_dir = f"outputs/diagnostics_phase2/{domain_name}"
    os.makedirs(output_dir, exist_ok=True)

    for conf in conf_thresholds:
        for iou_t in nms_thresholds:
            total_preds = 0
            total_matched = 0
            total_gt = 0

            for record in audit_records:
                gt_boxes_yxyx = torch.tensor(record["target"]["boxes"], dtype=torch.float32)
                total_gt += len(gt_boxes_yxyx)

                pred_boxes_yxyx, pred_scores, _ = decode_predictions_phase2(
                    record["obj_scores"], record["boxes_decoded"], record["cls_probs"],
                    conf_thresh=conf, iou_thresh=iou_t
                )
                
                total_preds += len(pred_boxes_yxyx)
                
                if len(pred_boxes_yxyx) > 0 and len(gt_boxes_yxyx) > 0:
                    pred_xyxy = convert_yxyx_to_xyxy(pred_boxes_yxyx)
                    gt_xyxy = convert_yxyx_to_xyxy(gt_boxes_yxyx)
                    
                    ious = box_iou(pred_xyxy, gt_xyxy)
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
    df.to_csv(f"{output_dir}/threshold_nms_sweep.csv", index=False)
    print(f"[{domain_name.upper()}] ✅ Updated Phase 2 Sweep Saved:\n", df.to_string())
    return df


def run_phase2_diagnostics():
    domains = ["server_room", "warehouse"]

    for domain in domains:
        print(f"\n==========================================")
        print(f"   RUNNING PHASE 2 DIAGNOSTICS: {domain.upper()}")
        print(f"==========================================")

        dataset = DynamicDetectionDataset(domain, "train")
        loader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=detection_collate_fn)

        num_classes = dataset.num_classes
        class_names = dataset.class_names

        print(f" -> Dynamically resolved metadata for {domain}:")
        print(f"    Classes ({num_classes}): {class_names}")

        model = FullViTGridDetector(domain=domain, num_classes=num_classes).to(device)

        collector = Phase2DiagnosticCollector(
            model=model,
            dataloader=loader,
            device=device,
            num_classes=num_classes,
            class_names=class_names,
            domain_name=domain
        )

        records = collector.collect_raw_and_decoded(num_samples=40)
        run_sweeps_phase2(records, domain)

    print("\n==========================================")
    print("  ✅ PHASE 2 COMPLETE FOR ALL DOMAINS")
    print("  Check 'outputs/diagnostics_phase2/'")
    print("==========================================")


if __name__ == "__main__":
    run_phase2_diagnostics()

