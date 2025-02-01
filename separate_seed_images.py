import argparse
import csv
import os
import shutil
from pathlib import Path
import random
from collections import defaultdict
import hashlib
from itertools import count


def parse_args():
    parser = argparse.ArgumentParser(
        description="Separate seed images based on germination data"
    )
    parser.add_argument("image_dir", help="Directory containing seed images")
    parser.add_argument("germination_data", help="Path to germination TSV file")
    parser.add_argument(
        "--undecided-frames",
        type=int,
        default=10,
        help="Number of frames after germination to mark as undecided (default: 10)",
    )
    parser.add_argument(
        "--outdir",
        required=True,
        help="Output directory for sorted images",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=10,
        help="Number of images to keep per seed for germinated and ungerminated categories",
    )
    return parser.parse_args()


def load_germination_data(tsv_path):
    germination_info = {}
    with open(tsv_path, "r") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            uid = row["UID"].split("_exp:")[0]  # Remove experiment name suffix
            germ_frame = row["Germination Detected on Frame"]
            if germ_frame.strip():  # Only store if there's a germination frame
                germination_info[uid] = int(germ_frame)
    return germination_info


def get_dir_hash(directory):
    """Generate a 4-character hash from directory path."""
    dir_hash = hashlib.md5(str(directory).encode()).hexdigest()
    return dir_hash[:4]


def process_images(image_dir, germination_data, undecided_frames, out_dir, max_samples):
    # Create output directories
    base_dir = Path(out_dir)
    base_dir.mkdir(exist_ok=True, parents=True)

    germinated_dir = base_dir / "Germinated"
    ungerminated_dir = base_dir / "Ungerminated"
    undecided_dir = base_dir / "Undecided"

    germinated_dir.mkdir(exist_ok=True)
    ungerminated_dir.mkdir(exist_ok=True)
    undecided_dir.mkdir(exist_ok=True)

    # Get hash for input directory
    dir_hash = get_dir_hash(image_dir)

    # Group images by seed and category
    seed_images = defaultdict(
        lambda: {"germinated": [], "ungerminated": [], "undecided": []}
    )

    # Create seed ID mapping
    seed_ids = {}
    next_id = count(1)

    # First pass: collect all unique seeds and assign sequential IDs
    for img_file in Path(image_dir).glob("roi_*"):
        parts = img_file.stem.split("_")
        if len(parts) < 5:
            continue

        uid = f"{parts[2]}_{parts[3]}_{parts[1]}"
        if uid not in germination_data:
            continue

        if uid not in seed_ids:
            seed_ids[uid] = next(next_id)

        # ...rest of grouping logic...
        frame_num = int(parts[4])
        germ_frame = germination_data[uid]

        if frame_num < germ_frame:
            seed_images[uid]["ungerminated"].append(img_file)
        elif frame_num < germ_frame + undecided_frames:
            seed_images[uid]["undecided"].append(img_file)
        else:
            seed_images[uid]["germinated"].append(img_file)

    # Process each seed's images
    for uid, categories in seed_images.items():
        seed_num = seed_ids[uid]

        # Process germinated and ungerminated
        for category in ["germinated", "ungerminated"]:
            images = categories[category]
            if images:
                selected = random.sample(images, min(max_samples, len(images)))
                dest_dir = (
                    germinated_dir if category == "germinated" else ungerminated_dir
                )
                for img_file in selected:
                    # Extract frame number from original filename parts
                    parts = img_file.stem.split("_")
                    frame_num = parts[4]  # Frame number is always at index 4
                    new_name = f"{dir_hash}{seed_num}_{frame_num}{img_file.suffix}"
                    shutil.copy(str(img_file), str(dest_dir / new_name))

        # Process undecided
        for img_file in categories["undecided"]:
            parts = img_file.stem.split("_")
            frame_num = parts[4]  # Frame number is always at index 4
            new_name = f"{dir_hash}{seed_num}_{frame_num}{img_file.suffix}"
            shutil.copy(str(img_file), str(undecided_dir / new_name))


def main():
    args = parse_args()
    germination_data = load_germination_data(args.germination_data)
    process_images(
        args.image_dir,
        germination_data,
        args.undecided_frames,
        args.outdir,
        args.samples,
    )


if __name__ == "__main__":
    main()
