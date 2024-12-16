import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple
from tqdm import tqdm
from PIL import Image


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
