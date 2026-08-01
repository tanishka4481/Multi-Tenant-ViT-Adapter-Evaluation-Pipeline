import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as scheduler
from transformers import ViTModel
from torch.utils.data import DataLoader
from torchvision.ops import sigmoid_focal_loss, complete_box_iou_loss

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"CELL 4 operating on active resource node: {device.upper()}")
GRID_SIZE = 14

# ----------------------------------------------------------------
# 1. HELPER & LOSS DEFINITIONS
# ----------------------------------------------------------------

def decode_cell_relative_boxes(box_preds, grid_size=GRID_SIZE):
    """
    Decodes cell-relative offsets [dx, dy, w, h] into global [ymin, xmin, ymax, xmax].
    Handles batched inputs (B, 196, 4) or unbatched (196, 4).
    """
    dev = box_preds.device
    rows, cols = torch.meshgrid(
        torch.arange(grid_size, device=dev),
        torch.arange(grid_size, device=dev),
        indexing='ij'
    )
    grid_y = rows.reshape(-1).float()
    grid_x = cols.reshape(-1).float()

    dx = box_preds[..., 0]
    dy = box_preds[..., 1]
    w  = box_preds[..., 2]
    h  = box_preds[..., 3]

    cy = (grid_y + dy) / grid_size
    cx = (grid_x + dx) / grid_size

    ymin = torch.clamp(cy - (h / 2.0), 0.0, 1.0)
    xmin = torch.clamp(cx - (w / 2.0), 0.0, 1.0)
    ymax = torch.clamp(cy + (h / 2.0), 0.0, 1.0)
    xmax = torch.clamp(cx + (w / 2.0), 0.0, 1.0)

    return torch.stack([ymin, xmin, ymax, xmax], dim=-1)


def compute_detection_loss(pred_obj, pred_box, pred_cls, 
                           obj_targets, box_targets, cls_targets, 
                           grid_size=GRID_SIZE):
    """
    Composite Focal + CIoU Loss for Grid Detection
    """
    obj_logits_flat = pred_obj.squeeze(-1) if pred_obj.dim() == 3 else pred_obj

    # 1. Focal Loss for Objectness
    loss_obj = sigmoid_focal_loss(
        obj_logits_flat,
        obj_targets,
        alpha=0.25,
        gamma=2.0,
        reduction='mean'
    )

    # 2. Mask positive cells
    pos_mask = (obj_targets > 0.5)
    num_pos = pos_mask.sum().item()

    if num_pos > 0:
        pred_boxes_xyxy = decode_cell_relative_boxes(pred_box, grid_size=grid_size)
        target_boxes_xyxy = decode_cell_relative_boxes(box_targets, grid_size=grid_size)

        pred_pos = pred_boxes_xyxy[pos_mask]
        target_pos = target_boxes_xyxy[pos_mask]

        # Convert [ymin, xmin, ymax, xmax] -> [xmin, ymin, xmax, ymax] for CIoU
        pred_pos_ciou = pred_pos[:, [1, 0, 3, 2]]
        target_pos_ciou = target_pos[:, [1, 0, 3, 2]]

        loss_box = complete_box_iou_loss(pred_pos_ciou, target_pos_ciou, reduction='mean')

        # 3. Cross Entropy for positive cell classes
        cls_logits_pos = pred_cls[pos_mask]
        cls_targets_pos = cls_targets[pos_mask]
        loss_cls = F.cross_entropy(cls_logits_pos, cls_targets_pos, reduction='mean')
    else:
        loss_box = torch.tensor(0.0, device=pred_obj.device)
        loss_cls = torch.tensor(0.0, device=pred_obj.device)
        
    return (3.0 * loss_obj) + (5.0 * loss_box) + (1.0 * loss_cls)


# ----------------------------------------------------------------
# 2. ARCHITECTURE HEAD DEFINITIONS
# ----------------------------------------------------------------

class PureRetailClassificationHead(nn.Module):
    """Classification head attached to the ViT [CLS] token."""
    def __init__(self, input_dim=768, num_classes=3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)


class GridDetectionHead(nn.Module):
    def __init__(self, input_dim=768, num_classes=2):
        super().__init__()
        self.obj_head = nn.Linear(input_dim, 1)
        self.box_head = nn.Linear(input_dim, 4)
        self.cls_head = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        obj_logits = self.obj_head(x)
        box_preds  = torch.sigmoid(self.box_head(x)) # Bound box predictions in [0, 1]
        cls_logits = self.cls_head(x)
        return obj_logits, box_preds, cls_logits


# ----------------------------------------------------------------
# 3. FROZEN ViT BACKBONE INITIALIZATION
# ----------------------------------------------------------------

backbone = ViTModel.from_pretrained("google/vit-base-patch16-224").to(device)
for param in backbone.parameters(): 
    param.requires_grad = False
backbone.eval()


# ----------------------------------------------------------------
# 4. TRAINING PROCEDURES
# ----------------------------------------------------------------

def train_retail_tenant(num_classes=3, epochs=5):
    print(f"\n--- Training Tuning Classification Model: [RETAIL] ---")
    dataset = IsolatedRetailDataset("train")
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    head = PureRetailClassificationHead(input_dim=768, num_classes=num_classes).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-4)

    # Compute inverse class frequencies to scale gradients during adapter training
    class_counts = torch.tensor([1395.0, 385.0, 176.0], device=device)
    weights = 1.0 / (class_counts / class_counts.sum())
    weights = weights / weights.sum() # Normalize
    
    criterion = nn.CrossEntropyLoss(weight=weights)
    
    for epoch in range(epochs):
        head.train()
        total_loss = 0.0
        for batch in loader:
            images = batch["pixel_values"].to(device)
            labels = batch["class_label"].to(device)
            
            optimizer.zero_grad()
            with torch.no_grad():
                cls_token = backbone(images).last_hidden_state[:, 0, :]
                
            outputs = head(cls_token)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} | Categorical Loss: {total_loss/len(loader):.4f}")

    os.makedirs("outputs/adapters", exist_ok=True)
    torch.save(head.state_dict(), "outputs/adapters/retail_head.pt")
    print(f"✅ Extracted optimized weights for [RETAIL].")


def train_detection_tenant(domain, epochs=10):
    print(f"\n--- Training High-Capacity Patch Detector: [{domain.upper()}] ---")
    
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
    train_retail_tenant(num_classes=3, epochs=8)
