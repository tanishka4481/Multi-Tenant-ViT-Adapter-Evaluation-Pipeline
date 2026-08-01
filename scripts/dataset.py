import os
import glob
import yaml
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import ViTImageProcessor

GRID_SIZE = 14

class MultiBoxDetectionDataset(Dataset):
    def __init__(self, domain_name, data_split="train", img_size=224):
        self.image_paths = []
        self.labels = []  
        self.img_size = img_size
        
        # Explicitly turn off internal resizing and center cropping to protect pixel alignment
        self.processor = ViTImageProcessor.from_pretrained(
            "google/vit-base-patch16-224",
            do_resize=False,
            do_center_crop=False
        )

        base_path = f"data/{domain_name}/{data_split}"
        img_dir = os.path.join(base_path, "images")
        lbl_dir = os.path.join(base_path, "labels")

        for img_p in glob.glob(os.path.join(img_dir, "*.*")):
            base_name = os.path.splitext(os.path.basename(img_p))[0]
            lbl_p = os.path.join(lbl_dir, f"{base_name}.txt")

            boxes, classes = [], []
            if os.path.exists(lbl_p):
                with open(lbl_p, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) != 5:
                            continue
                        cls_idx = int(parts[0])
                        cx, cy, w, h = map(float, parts[1:])
                        
                        # Direct normalized bounding boxes matching a perfect square resize mapping
                        ymin = max(0.0, cy - h / 2)
                        xmin = max(0.0, cx - w / 2)
                        ymax = min(1.0, cy + h / 2)
                        xmax = min(1.0, cx + w / 2)
                        
                        boxes.append([ymin, xmin, ymax, xmax])
                        classes.append(cls_idx)

            self.image_paths.append(img_p)
            self.labels.append({"boxes": boxes, "classes": classes})

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            # Force a direct layout stretch resize so coordinates preserve relative structural scale
            img = Image.open(self.image_paths[idx]).convert("RGB").resize((self.img_size, self.img_size))
        except Exception:
            img = Image.new("RGB", (self.img_size, self.img_size), (0, 0, 0))

        pixel_values = self.processor(images=img, return_tensors="pt")["pixel_values"].squeeze(0)
        return {
            "pixel_values": pixel_values,
            "boxes": self.labels[idx]["boxes"],       
            "classes": self.labels[idx]["classes"]
        }


def detection_collate_fn(batch):
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "boxes": [b["boxes"] for b in batch],
        "classes": [b["classes"] for b in batch]
    }


class IsolatedRetailDataset(Dataset):
    def __init__(self, data_split="train", img_size=224):
        self.image_paths = []
        self.class_labels = []
        self.img_size = img_size
        self.processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224", do_resize=False, do_center_crop=False)
        
        base_path = f"data/retail/{data_split}"
        class_folders = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
        self.class_to_idx = {name.lower(): idx for idx, name in enumerate(sorted(class_folders))}
        
        for cls_name in class_folders:
            cls_idx = self.class_to_idx[cls_name.lower()]
            for img_p in glob.glob(os.path.join(base_path, cls_name, "*.*")):
                self.image_paths.append(img_p)
                self.class_labels.append(cls_idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.image_paths[idx]).convert("RGB").resize((self.img_size, self.img_size))
        except Exception:
            img = Image.new("RGB", (self.img_size, self.img_size), (0, 0, 0))
        pixel_values = self.processor(images=img, return_tensors="pt")["pixel_values"].squeeze(0)
        return {"pixel_values": pixel_values, "class_label": torch.tensor(self.class_labels[idx], dtype=torch.long)}


class DynamicDetectionDataset(Dataset):
    """
    Detection Dataset that dynamically parses data.yaml to adapt 
    to class counts and label mappings automatically.
    """
    def __init__(self, domain, data_split="train", img_size=224):
        self.image_paths = []
        self.labels = []
        self.img_size = img_size
        self.domain = domain
        
        self.processor = ViTImageProcessor.from_pretrained(
            "google/vit-base-patch16-224", 
            do_resize=False, 
            do_center_crop=False
        )
        
        base_dir = f"data/{domain}"
        yaml_path = os.path.join(base_dir, "data.yaml")
        
        # Parse data.yaml for class metadata
        if os.path.exists(yaml_path):
            with open(yaml_path, "r") as f:
                meta = yaml.safe_load(f)
            self.class_names = meta.get("names", [])
            self.num_classes = meta.get("nc", len(self.class_names))
        else:
            raise FileNotFoundError(f"Could not find data.yaml in {base_dir}")

        img_dir = os.path.join(base_dir, data_split, "images")
        lbl_dir = os.path.join(base_dir, data_split, "labels")

        for img_p in sorted(glob.glob(os.path.join(img_dir, "*.*"))):
            base_name = os.path.splitext(os.path.basename(img_p))[0]
            lbl_p = os.path.join(lbl_dir, f"{base_name}.txt")

            boxes = []
            classes = []

            if os.path.exists(lbl_p):
                with open(lbl_p, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) != 5:
                            continue
                        cls_idx = int(parts[0])
                        
                        # Guard against out-of-range class indices
                        if cls_idx >= self.num_classes:
                            continue

                        cx, cy, w, h = map(float, parts[1:])

                        # Convert YOLO center format (cx, cy, w, h) -> Normalized (ymin, xmin, ymax, xmax)
                        ymin = max(0.0, cy - h / 2.0)
                        xmin = max(0.0, cx - w / 2.0)
                        ymax = min(1.0, cy + h / 2.0)
                        xmax = min(1.0, cx + w / 2.0)

                        boxes.append([ymin, xmin, ymax, xmax])
                        classes.append(cls_idx)

            self.image_paths.append(img_p)
            self.labels.append({"boxes": boxes, "classes": classes})

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.image_paths[idx]).convert("RGB").resize((self.img_size, self.img_size))
        except Exception:
            img = Image.new("RGB", (self.img_size, self.img_size), (0, 0, 0))

        pixel_values = self.processor(images=img, return_tensors="pt")["pixel_values"].squeeze(0)
        
        return {
            "pixel_values": pixel_values,
            "boxes": self.labels[idx]["boxes"],
            "classes": self.labels[idx]["classes"]
        }


def assign_targets_to_grid(boxes, classes, grid_size=GRID_SIZE):
    """
    Encodes global normalized targets [ymin, xmin, ymax, xmax] 
    into cell-relative offset targets (dx, dy, w, h) for GridDetectionHead.
    """
    obj_target = torch.zeros(grid_size * grid_size)
    box_target = torch.zeros(grid_size * grid_size, 4)
    cls_target = torch.zeros(grid_size * grid_size, dtype=torch.long)
    
    for box, cls in zip(boxes, classes):
        if isinstance(box, torch.Tensor):
            box = box.cpu().tolist()
            
        ymin, xmin, ymax, xmax = box
        
        # Calculate box center and absolute dimensions
        cy = (ymin + ymax) / 2.0
        cx = (xmin + xmax) / 2.0
        w = max(xmax - xmin, 1e-4)
        h = max(ymax - ymin, 1e-4)
        
        # Determine grid cell row and col
        row = min(int(cy * grid_size), grid_size - 1)
        col = min(int(cx * grid_size), grid_size - 1)
        cell_idx = row * grid_size + col
        
        # Calculate cell-relative offsets in range [0, 1]
        dy = (cy * grid_size) - row
        dx = (cx * grid_size) - col
        
        obj_target[cell_idx] = 1.0
        box_target[cell_idx] = torch.tensor([dx, dy, w, h], dtype=torch.float32)
        cls_target[cell_idx] = cls if isinstance(cls, torch.Tensor) else torch.tensor(cls, dtype=torch.long)
        
    return obj_target, box_target, cls_target
