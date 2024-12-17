import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from ml_utils import split_by_seed_id


class SpiroCropDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


def create_dataset(base_path, split="train", transform=None):
    """
    Create dataset with train/val/test splits.

    Args:
        base_path: Path to the dataset directory
        split: One of 'train', 'val', 'test'
        transform: Optional transforms to apply
    """
    all_paths = []
    all_labels = []
    all_seed_ids = []

    # Collect paths, labels, and seed_ids
    for root, _, files in os.walk(base_path):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".png"):
                path = os.path.join(root, file)
                label = os.path.basename(root)
                seed_id = int(file.split("_")[0])
                all_paths.append(path)
                all_labels.append(label)
                all_seed_ids.append(seed_id)

    # Get train/val/test split indices
    train_ids, val_ids, test_ids = split_by_seed_id(np.array(all_seed_ids))

    # Select appropriate seed IDs based on split
    if split == "train":
        valid_seeds = train_ids
    elif split == "val":
        valid_seeds = val_ids
    else:  # test
        valid_seeds = test_ids

    # Filter data based on split
    mask = np.isin(all_seed_ids, valid_seeds)
    filtered_paths = [p for i, p in enumerate(all_paths) if mask[i]]
    filtered_labels = [l for i, l in enumerate(all_labels) if mask[i]]

    return SpiroCropDataset(filtered_paths, filtered_labels, transform)
