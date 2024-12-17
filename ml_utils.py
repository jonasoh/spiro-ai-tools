import numpy as np


def split_by_seed_id(seed_ids, train_ratio=0.7, val_ratio=0.2, random_state=42):
    """
    Split seed IDs into train, validation, and test sets.

    Args:
        seed_ids: Array of unique seed IDs
        train_ratio: Proportion for training (default: 0.7)
        val_ratio: Proportion for validation (default: 0.2)
        random_state: Random seed for reproducibility

    Returns:
        train_ids, val_ids, test_ids: Arrays of seed IDs for each split
    """
    unique_seeds = np.unique(seed_ids)
    np.random.seed(random_state)
    np.random.shuffle(unique_seeds)

    n_seeds = len(unique_seeds)
    train_size = int(n_seeds * train_ratio)
    val_size = int(n_seeds * val_ratio)

    train_ids = unique_seeds[:train_size]
    val_ids = unique_seeds[train_size : train_size + val_size]
    test_ids = unique_seeds[train_size + val_size :]

    return train_ids, val_ids, test_ids
