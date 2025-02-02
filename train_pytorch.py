import os
import torch
import argparse
import torchvision
from PIL import Image
from pycocotools.coco import COCO
from torchvision.transforms import ToTensor
from tqdm import tqdm
import numpy as np


class SeedDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, ann_file):
        self.img_dir = img_dir
        self.coco = COCO(ann_file)
        self.img_ids = list(self.coco.imgs.keys())
        self.transform = ToTensor()

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        ann_ids = self.coco.getAnnIds(img_id)
        anns = self.coco.loadAnns(ann_ids)
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_dir, img_info["file_name"])
        img = Image.open(img_path).convert("RGB")
        boxes = []
        masks = []
        labels = []
        for a in anns:
            bbox = a["bbox"]
            boxes.append(bbox)
            masks.append(self.coco.annToMask(a))
            labels.append(a["category_id"])
        boxes_xywh = np.array(boxes, dtype=np.float32)
        boxes_x1y1x2y2 = boxes_xywh.copy()
        boxes_x1y1x2y2[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2]  # xmax = x + w
        boxes_x1y1x2y2[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3]  # ymax = y + h
        boxes_x1y1x2y2[:, 0] = boxes_xywh[:, 0]  # xmin = x
        boxes_x1y1x2y2[:, 1] = boxes_xywh[:, 1]  # ymin = y
        boxes = torch.as_tensor(boxes_x1y1x2y2, dtype=torch.float32)

        masks = torch.as_tensor(np.stack(masks, axis=0), dtype=torch.uint8)
        labels = torch.as_tensor(np.array(labels, dtype=np.int64))
        return self.transform(img), dict(boxes=boxes, labels=labels, masks=masks)

    def __len__(self):
        return len(self.img_ids)


def collate_fn(batch):
    return tuple(zip(*batch))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Mask R-CNN model for seed detection"
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="number of epochs to train"
    )
    parser.add_argument(
        "--batch-size", type=int, default=4, help="batch size for training"
    )
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument(
        "--num-workers", type=int, default=4, help="number of worker processes"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="model_weights.pth",
        help="output path for final model weights",
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default=".", help="directory to save checkpoints"
    )
    parser.add_argument(
        "--weight-decay", type=float, default=0.0001, help="weight decay for optimizer"
    )
    parser.add_argument(
        "--scheduler-patience",
        type=int,
        default=2,
        help="patience for learning rate scheduler",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="directory containing the image data",
    )
    parser.add_argument(
        "--train-annotations",
        type=str,
        required=True,
        help="path to train COCO annotations file",
    )
    parser.add_argument(
        "--val-annotations",
        type=str,
        required=True,
        help="path to validation COCO annotations file",
    )
    args = parser.parse_args()

    device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load datasets with validation
    print("Loading training dataset...")
    train_dataset = SeedDataset(args.data_dir, args.train_annotations)
    if len(train_dataset) == 0:
        raise ValueError(
            f"Training dataset is empty! Please check the paths:\n"
            f"Data directory: {args.data_dir}\n"
            f"Train annotations: {args.train_annotations}"
        )
    print(f"Training dataset size: {len(train_dataset)}")

    print("Loading validation dataset...")
    val_dataset = SeedDataset(args.data_dir, args.val_annotations)
    if len(val_dataset) == 0:
        raise ValueError(
            f"Validation dataset is empty! Please check the paths:\n"
            f"Data directory: {args.data_dir}\n"
            f"Validation annotations: {args.val_annotations}"
        )
    print(f"Validation dataset size: {len(val_dataset)}")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
        prefetch_factor=2,
        collate_fn=collate_fn,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        prefetch_factor=2,
        collate_fn=collate_fn,
    )

    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
    model.roi_heads.box_predictor = (
        torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
            model.roi_heads.box_predictor.cls_score.in_features,
            3,  # background + 2 seed classes
        )
    )
    model.roi_heads.mask_predictor = (
        torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
            model.roi_heads.mask_predictor.conv5_mask.in_channels, 256, 3
        )
    )
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.1,
        patience=args.scheduler_patience,
        verbose=True,
    )

    # track best loss for model saving
    best_loss = float("inf")
    checkpoint_path = os.path.join(args.checkpoint_dir, "best_model.pth")
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    for epoch in range(args.epochs):
        # Training phase
        total_loss = 0.0
        model.train()
        for images, targets in tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{args.epochs} - Training"
        ):
            optimizer.zero_grad()
            losses = model(images, targets)
            loss = sum(losses.values())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc="Validation"):
                images = [image.to(device) for image in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                losses = model(images, targets)
                val_loss += sum(losses.values()).item()

        avg_val_loss = val_loss / len(val_loader)
        print(
            f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
        )

        # Update learning rate based on validation loss
        scheduler.step(avg_val_loss)

        # Save best model based on validation loss
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": best_loss,
                },
                checkpoint_path,
            )

    # save final model with specified output path
    torch.save(model.state_dict(), args.output)
    print(f"Training completed. Model saved to {args.output}")
