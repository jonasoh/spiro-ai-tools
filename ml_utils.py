import os
import cv2
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple
from tqdm import tqdm
from PIL import Image


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


def save_image(save_info: Tuple[Image.Image, str, dict]) -> str:
    """Save a single image and return its path."""
    img, path, params = save_info
    img.save(path, **params)
    return path


def parallel_save_images(
    images: List[Image.Image],
    paths: List[str],
    save_params: dict,
    max_workers: int = None,
) -> None:
    """
    Save images in parallel while showing progress.

    Args:
        images: List of PIL Image objects
        paths: List of paths where to save the images
        save_params: Parameters for image saving (format, quality, etc.)
        max_workers: Maximum number of parallel processes
    """
    if max_workers is None:
        max_workers = os.cpu_count() or 1

    save_infos = [(img, path, save_params) for img, path in zip(images, paths)]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(save_image, info) for info in save_infos]

        with tqdm(total=len(futures), desc="Saving images") as pbar:
            for future in as_completed(futures):
                future.result()  # Get the result to ensure no exceptions were raised
                pbar.update(1)


def get_connected_object(binary_image, x, y, min_size=0):
    """
    Extracts the connected component in a binary image that includes a specified point (x, y).
    Optionally filters the object by minimum size.

    Parameters:
        binary_image (numpy.ndarray): Binary image (grayscale) where objects are white (255) and background is black (0).
        x (int): X-coordinate of the point.
        y (int): Y-coordinate of the point.
        min_size (int): Minimum size (in pixels) for the connected component to be kept. Default is 0.

    Returns:
        numpy.ndarray: Binary image containing only the connected component, or an empty mask if no valid object is found.
    """
    if not (0 <= x < binary_image.shape[1] and 0 <= y < binary_image.shape[0]):
        raise ValueError(f"Point ({x}, {y}) is outside the image bounds.")

    if binary_image[y, x] == 0:
        raise ValueError(f"Point ({x}, {y}) does not belong to any object.")

    mask = np.zeros(
        (binary_image.shape[0] + 2, binary_image.shape[1] + 2), dtype=np.uint8
    )
    flood_filled_image = binary_image.copy()
    _, _, _, rect = cv2.floodFill(
        flood_filled_image, mask, seedPoint=(x, y), newVal=128
    )

    connected_component = (flood_filled_image == 128).astype(np.uint8) * 255

    if min_size > 0:
        contours, _ = cv2.findContours(
            connected_component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        largest_contour = max(contours, key=cv2.contourArea) if contours else None

        if largest_contour and cv2.contourArea(largest_contour) >= min_size:
            filtered_image = np.zeros_like(binary_image)
            cv2.drawContours(
                filtered_image, [largest_contour], -1, 255, thickness=cv2.FILLED
            )
            return filtered_image
        else:
            return np.zeros_like(binary_image)

    return connected_component


def get_objects_from_coordinates(binary_image, coordinates, verbose=False):
    """
    Keeps objects connected to specific coordinates in a binary image.

    Parameters:
        binary_image (numpy.ndarray): Binary image (grayscale), where objects are white (255) and background is black (0).
        coordinates (list of tuples): List of (x, y) coordinates.
        verbose (bool): Whether to print debug information.

    Returns:
        numpy.ndarray: Binary mask containing only the objects connected to the specified coordinates.
    """
    final_mask = np.zeros_like(binary_image, dtype=np.uint8)
    processed_image = binary_image.copy()

    for x, y in coordinates:
        if not (
            0 <= x < processed_image.shape[1] and 0 <= y < processed_image.shape[0]
        ):
            if verbose:
                print(f"Skipping ({x}, {y}): point is outside image bounds.")
            continue

        if processed_image[y, x] == 0:
            if verbose:
                print(f"Skipping ({x}, {y}): point does not belong to any object.")
            continue

        mask = np.zeros(
            (processed_image.shape[0] + 2, processed_image.shape[1] + 2), dtype=np.uint8
        )
        flood_filled_image = processed_image.copy()
        cv2.floodFill(flood_filled_image, mask, seedPoint=(x, y), newVal=128)

        final_mask[flood_filled_image == 128] = 255
        processed_image[flood_filled_image == 128] = 0

    return final_mask
