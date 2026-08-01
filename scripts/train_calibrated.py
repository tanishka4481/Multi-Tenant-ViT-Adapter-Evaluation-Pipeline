import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as scheduler
from torch.utils.data import DataLoader

GRID_SIZE = 14
device = "cuda" if torch.cuda.is_available() else "cpu"

class SigmoidFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)
        
        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss
            
        return loss.sum()


def compute_detection_loss(pred_obj, pred_box, pred_cls, obj_targets, box_targets, cls_targets, grid_size=14):
    if pred_obj.dim() == 3:
        pred_obj = pred_obj.squeeze(-1)
        
    pos_weight = torch.tensor([15.0], device=pred_obj.device)
    bce_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="mean")
    loss_obj = bce_fn(pred_obj, obj_targets)
    
    pos_mask = (obj_targets > 0.5)
    num_pos = pos_mask.sum().item() + 1e-6
    
    if pos_mask.sum() > 0:
        loss_box = F.l1_loss(pred_box[pos_mask], box_targets[pos_mask], reduction="sum") / num_pos
        loss_cls = F.cross_entropy(pred_cls[pos_mask], cls_targets[pos_mask], reduction="sum") / num_pos
    else:
        loss_box = torch.tensor(0.0, device=pred_obj.device)
        loss_cls = torch.tensor(0.0, device=pred_obj.device)
        
    total_loss = (2.0 * loss_obj) + (3.0 * loss_box) + (1.0 * loss_cls)
    return total_loss


def train_detection_tenant(domain, epochs=20):
    print(f"\n--- Training High-Capacity Patch Detector (Calibrated): [{domain.upper()}] ---")
    
    dataset = DynamicDetectionDataset(domain, "train")
    loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=detection_collate_fn)
    
    num_classes = dataset.num_classes
    print(f" -> Configured classes for {domain}: {dataset.class_names} (Count: {num_classes})")
    
    head = GridDetectionHead(input_dim=768, num_classes=num_classes).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    lr_scheduler = scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    for epoch in range(epochs):
        head.train()
        total_loss = 0.0
        for batch in loader:
            optimizer.zero_grad()
            pixels = batch["pixel_values"].to(device)
            batch_size = pixels.size(0)
            
            with torch.no_grad():
                patch_tokens = backbone(pixels).last_hidden_state[:, 1:, :]
                
            pred_obj, pred_box, pred_cls = head(patch_tokens)
            
            obj_targets_list, box_targets_list, cls_targets_list = [], [], []
            for i in range(batch_size):
                obj_t, box_t, cls_t = assign_targets_to_grid(batch["boxes"][i], batch["classes"][i])
                obj_targets_list.append(obj_t)
                box_targets_list.append(box_t)
                cls_targets_list.append(cls_t)
                
            obj_targets = torch.stack(obj_targets_list).to(device)
            box_targets = torch.stack(box_targets_list).to(device)
            cls_targets = torch.stack(cls_targets_list).to(device)
            
            loss = compute_detection_loss(
                pred_obj, pred_box, pred_cls,
                obj_targets, box_targets, cls_targets,
                grid_size=GRID_SIZE
            )
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        lr_scheduler.step()
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(loader):.4f} | Current LR: {optimizer.param_groups[0]['lr']:.6f}")
        
    os.makedirs("outputs/adapters", exist_ok=True)
    torch.save(head.state_dict(), f"outputs/adapters/{domain}_head.pt")
    print(f"✅ Extracted high-capacity weights for [{domain}].")


if __name__ == "__main__":
    train_detection_tenant("server_room", epochs=20)
    train_detection_tenant("warehouse", epochs=20)
