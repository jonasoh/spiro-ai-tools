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
    args = parser.parse_args()

    device = torch.device("cpu")
    print(f"Using device: {device}")

    data_loader = torch.utils.data.DataLoader(
        SeedDataset("roboflow_batch1/", "roboflow_batch1/train_coco.json"),
        batch_size=args.batch_size,
        shuffle=True,
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

    for epoch in range(args.epochs):
        total_loss = 0.0
        model.train()

        for images, targets in tqdm(data_loader):
            optimizer.zero_grad()

            # simple forward pass without autocast
            losses = model(images, targets)
            loss = sum(losses.values())

            # regular backward pass
            loss.backward()

            # clip gradients for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(data_loader)
        print(f"epoch {epoch}: avg loss: {avg_loss:.4f}")

        # update learning rate based on loss
        scheduler.step(avg_loss)

        # save best model with args.checkpoint_dir
        if avg_loss < best_loss:
            best_loss = avg_loss
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
