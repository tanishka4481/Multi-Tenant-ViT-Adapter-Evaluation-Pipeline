import os
import json
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision

print("Launching Clean Global Evaluation Engine...")

GRID_SIZE = 14
device = "cuda" if torch.cuda.is_available() else "cpu"

def evaluate_detection_tenant(domain, num_classes):
    print(f"\nCalculating true mAP metrics for: [{domain.upper()}]...")
    dataset = MultiBoxDetectionDataset(domain, "test")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=detection_collate_fn)
    
    head = GridDetectionHead(input_dim=768, num_classes=num_classes).to(device)
    head.load_state_dict(torch.load(f"outputs/adapters/{domain}_head.pt", map_location=device))
    head.eval()
    
    metric = MeanAveragePrecision(iou_type="bbox")
    valid_eval_counts = 0
    
    for batch in tqdm(loader):
        if not batch["boxes"] or len(batch["boxes"]) == 0 or len(batch["boxes"][0]) == 0:
            continue 
            
        pixels = batch["pixel_values"].to(device)
        raw_boxes = batch["boxes"][0]
        raw_classes = batch["classes"][0]
        
        with torch.no_grad():
            patch_tokens = backbone(pixels).last_hidden_state[:, 1:, :] 
            pred_obj, pred_box, pred_cls = head(patch_tokens)
            
        pred_obj = torch.sigmoid(pred_obj[0].squeeze(-1))      # Shape: [196]
        pred_box = pred_box[0]                                  # Shape: [196, 4]
        pred_cls = torch.softmax(pred_cls[0], dim=-1)           # Shape: [196, C]
        
        valid_indices = torch.where(pred_obj > 0.25)[0]
        
        p_boxes_list = []
        p_scores_list = []
        p_labels_list = []
        
        for idx in valid_indices:
            ymin, xmin, ymax, xmax = pred_box[idx]
            
            p_boxes_list.append([
                ymin.item() * 224.0, 
                xmin.item() * 224.0, 
                ymax.item() * 224.0, 
                xmax.item() * 224.0
            ])
            p_scores_list.append(pred_obj[idx].item())
            p_labels_list.append(torch.argmax(pred_cls[idx]).item())
            
        if len(p_boxes_list) > 0:
            p_boxes = torch.tensor(p_boxes_list, dtype=torch.float32)
            p_scores = torch.tensor(p_scores_list, dtype=torch.float32)
            p_labels = torch.tensor(p_labels_list, dtype=torch.int64)
        else:
            p_boxes = torch.zeros((0, 4))
            p_scores = torch.zeros((0,))
            p_labels = torch.zeros((0,), dtype=torch.long)
            
        t_boxes = torch.tensor(raw_boxes, dtype=torch.float32) * 224.0
        t_labels = torch.tensor(raw_classes, dtype=torch.int64)
        
        metric.update(
            [{"boxes": p_boxes, "scores": p_scores, "labels": p_labels}],
            [{"boxes": t_boxes, "labels": t_labels}]
        )
        valid_eval_counts += 1
        
    if valid_eval_counts == 0:
        return 0.0
        
    results = metric.compute()
    calculated_map = round(results["map_50"].item(), 3)
    
    del head
    torch.cuda.empty_cache()
    return max(0.0, calculated_map)

# Run direct global coordinate lookup evaluations
eval_summary = {
    "retail": {"accuracy": evaluate_retail_tenant()},
    "server_room": {"mAP_0_5": evaluate_detection_tenant("server_room", 3)},
    "warehouse": {"mAP_0_5": evaluate_detection_tenant("warehouse", 2)}
}

print("\n==========================================")
print("       VERIFIED ENGINEERING METRICS       ")
print("==========================================")
print(json.dumps(eval_summary, indent=4))
print("==========================================")
