import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

def run_numerical_and_visual_audit():
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # ----------------------------------------------------------------
    # 1. AUDIT WAREHOUSE (Detection Domain)
    # ----------------------------------------------------------------
    try:
        wh_dataset = DynamicDetectionDataset(domain="warehouse", data_split="train")
        wh_loader = DataLoader(wh_dataset, batch_size=1, shuffle=True, collate_fn=detection_collate_fn)
        wh_batch = next(iter(wh_loader))
        
        wh_pixels = wh_batch["pixel_values"][0]
        wh_boxes = wh_batch["boxes"][0]
        wh_classes = wh_batch["classes"][0]
        
        obj_t, box_t, cls_t = assign_targets_to_grid(wh_boxes, wh_classes)
        active_cells = int(obj_t.sum().item())
        
        print("==============================================================")
        print("--- [1/3] WAREHOUSE DETECTION TENSOR STREAM ---")
        print(f" -> Pixel Tensor Shape: {list(wh_pixels.shape)}")
        print(f" -> Target Grid Active Object Cells: {active_cells} / 196")
        print(f" -> Classes defined in YAML: {wh_dataset.class_names}")
        
        img_np = wh_pixels.permute(1, 2, 0).cpu().numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-6)
        axes[0].imshow(img_np)
        axes[0].set_title(f"Warehouse (Active Cells: {active_cells})")
        
        for box, cls_i in zip(wh_boxes, wh_classes):
            ymin, xmin, ymax, xmax = box
            rect = plt.Rectangle((xmin * 224, ymin * 224), (xmax - xmin) * 224, (ymax - ymin) * 224, 
                                 fill=False, color="lime", linewidth=2)
            axes[0].add_patch(rect)
            label_text = wh_dataset.class_names[cls_i] if cls_i < len(wh_dataset.class_names) else str(cls_i)
            axes[0].text(xmin * 224, ymin * 224, label_text, color="black", backgroundcolor="lime", fontsize=8)
            
    except Exception as e:
        print(f"❌ Warehouse Audit Failed: {str(e)}")
        axes[0].text(0.1, 0.5, f"Warehouse Error:\n{str(e)}", color="red")

    # ----------------------------------------------------------------
    # 2. AUDIT SERVER ROOM (Updated 2-Class Detection Domain)
    # ----------------------------------------------------------------
    try:
        sr_dataset = DynamicDetectionDataset(domain="server_room", data_split="train")
        sr_loader = DataLoader(sr_dataset, batch_size=1, shuffle=True, collate_fn=detection_collate_fn)
        sr_batch = next(iter(sr_loader))
        
        sr_pixels = sr_batch["pixel_values"][0]
        sr_boxes = sr_batch["boxes"][0]
        sr_classes = sr_batch["classes"][0]
        
        obj_t, box_t, cls_t = assign_targets_to_grid(sr_boxes, sr_classes)
        active_cells = int(obj_t.sum().item())
        
        print("-" * 60)
        print("--- [2/3] SERVER ROOM DETECTION TENSOR STREAM ---")
        print(f" -> Pixel Tensor Shape: {list(sr_pixels.shape)}")
        print(f" -> Target Grid Active Object Cells: {active_cells} / 196")
        print(f" -> Detected Classes: {sr_dataset.class_names}")
        
        for idx in torch.where(obj_t == 1.0)[0]:
            cls_idx_item = cls_t[idx].item()
            cls_str = sr_dataset.class_names[cls_idx_item] if cls_idx_item < len(sr_dataset.class_names) else str(cls_idx_item)
            print(f"    ↳ Cell {idx} Mapped Class ID: {cls_idx_item} ({cls_str})")
        print("-" * 60)
        
        img_np = sr_pixels.permute(1, 2, 0).cpu().numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-6)
        axes[1].imshow(img_np)
        axes[1].set_title(f"Server Room (Active Cells: {active_cells})")
        
        for box, cls_i in zip(sr_boxes, sr_classes):
            ymin, xmin, ymax, xmax = box
            rect = plt.Rectangle((xmin * 224, ymin * 224), (xmax - xmin) * 224, (ymax - ymin) * 224, 
                                 fill=False, color="orange", linewidth=2)
            axes[1].add_patch(rect)
            label_text = sr_dataset.class_names[cls_i] if cls_i < len(sr_dataset.class_names) else str(cls_i)
            axes[1].text(xmin * 224, ymin * 224, label_text, color="black", backgroundcolor="orange", fontsize=8)
            
    except Exception as e:
        print(f"❌ Server Room Audit Failed: {str(e)}")
        axes[1].text(0.1, 0.5, f"Server Room Error:\n{str(e)}", color="red")

    # ----------------------------------------------------------------
    # 3. AUDIT RETAIL (Isolated Classification Domain)
    # ----------------------------------------------------------------
    try:
        rt_dataset = IsolatedRetailDataset("train")
        rt_loader = DataLoader(rt_dataset, batch_size=1, shuffle=True)
        rt_batch = next(iter(rt_loader))
        
        rt_pixels = rt_batch["pixel_values"][0]
        rt_label = rt_batch["class_label"][0]
        
        inv_map = {v: k for k, v in rt_dataset.class_to_idx.items()}
        cls_name = inv_map.get(rt_label.item(), "unknown").upper()
        
        print("--- [3/3] RETAIL CLASSIFICATION TENSOR STREAM ---")
        print(f" -> Pixel Tensor Shape entering model: {list(rt_pixels.shape)}")
        print(f" -> Target Scalar Class ID: {rt_label.item()} Label: ({cls_name})")
        print("==============================================================")
        
        img_np = rt_pixels.permute(1, 2, 0).cpu().numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-6)
        axes[2].imshow(img_np)
        axes[2].set_title(f"Retail Target: {cls_name}")
        
    except Exception as e:
        print(f"❌ Retail Audit Failed: {str(e)}")
        axes[2].text(0.1, 0.5, f"Retail Error:\n{str(e)}", color="red")

    for ax in axes[:2]:
        ax.set_xlim(0, 224)
        ax.set_ylim(224, 0)
        ax.axis("on")
    axes[2].axis("off")
    
    plt.suptitle("End-to-End Multi-Tenant Pipeline Tensor Verification", fontsize=16, weight="bold")
    plt.tight_layout()
    plt.show()

# Run the live pipeline visualization sweep
run_numerical_and_visual_audit()
