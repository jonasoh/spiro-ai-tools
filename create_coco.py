import os
import cv2
import json
import argparse
from ml_utils import split_by_seed_id


def create_coco_json(data_dir, output_json, valid_seed_ids=None):
    """
    Create COCO JSON with optional seed_id filtering

    Args:
        data_dir: Directory containing the image data
        output_json: Path to save the COCO JSON
        valid_seed_ids: Optional list of seed IDs to include
    """
    coco_format = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "germinated", "supercategory": "seed"},
            {"id": 2, "name": "ungerminated", "supercategory": "seed"},
        ],
    }
    annotation_id = image_id = 1

    # Collect all seed IDs first if we need to split
    all_seed_ids = set()
    for experiment in os.listdir(data_dir):
        if not os.path.isdir(experiment_path := os.path.join(data_dir, experiment)):
            continue
        for category in ["germinated", "ungerminated"]:
            category_path = os.path.join(experiment_path, category)
            if not os.path.isdir(category_path):
                continue
            for file in os.listdir(category_path):
                if not file.endswith("_crop.png"):
                    continue
                # Update seed_id extraction
                parts = file.replace("_crop.png", "").split("_")
                roi_num = parts[1]  # e.g., "13"
                plate = parts[2]  # e.g., "plate3"
                exp_name = parts[3]  # e.g., "ALL"
                seed_id = f"{roi_num}_{plate}_{exp_name}"
                all_seed_ids.add(seed_id)

    for experiment in os.listdir(data_dir):
        if not os.path.isdir(experiment_path := os.path.join(data_dir, experiment)):
            continue

        for category in ["germinated", "ungerminated"]:
            if not os.path.isdir(
                category_path := os.path.join(experiment_path, category)
            ):
                continue

            category_id = 1 if category == "germinated" else 2

            for file in os.listdir(category_path):
                if not file.endswith("_crop.png"):
                    continue  # only process crop images

                # extract metadata from file name
                base_name = file.replace("_crop.png", "")
                parts = base_name.split("_")
                roi_num = parts[1]  # e.g., "23"
                plate = parts[2]  # e.g., "plate3"
                group = parts[3]  # e.g., "ALL"
                seed_id = f"{roi_num}_{plate}_{group}_{experiment.replace(' ', '_')}"
                slice_num = int(parts[-1])

                crop_file = os.path.join(category_path, file)
                mask_file = os.path.join(category_path, f"{base_name}_mask.png")

                if not os.path.exists(mask_file):
                    print(mask_file)
                    print(f"Warning: Missing mask for {file}")
                    continue

                # read image and mask
                image = cv2.imread(crop_file)
                mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
                height, width, _ = image.shape
                # Only include if no filter or seed_id is in valid_seed_ids
                if valid_seed_ids is None or seed_id in valid_seed_ids:
                    # add image metadata
                    coco_format["images"].append(
                        {
                            "id": image_id,
                            "file_name": f"{experiment}/{category}/{file}",
                            "width": width,
                            "height": height,
                            "experiment": experiment,  # include experiment name
                            "seed_id": seed_id,
                            "slice": slice_num,
                        }
                    )

                    # process mask for contours
                    contours, _ = cv2.findContours(
                        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    for contour in contours:
                        if len(contour) < 3:  # skip invalid contours
                            continue

                        segmentation = contour.flatten().tolist()
                        x, y, w, h = cv2.boundingRect(contour)
                        area = cv2.contourArea(contour)

                        # add annotation
                        coco_format["annotations"].append(
                            {
                                "id": annotation_id,
                                "image_id": image_id,
                                "category_id": category_id,
                                "segmentation": [segmentation],
                                "bbox": [x, y, w, h],
                                "area": area,
                                "iscrowd": 0,
                                "seed_id": seed_id,  # include seed ID
                            }
                        )
                        annotation_id += 1

                    image_id += 1

    # save COCO JSON
    with open(output_json, "w") as f:
        json.dump(coco_format, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate COCO JSON from training data"
    )
    parser.add_argument(
        "--dir", type=str, required=True, help="Path to training data folder"
    )
    parser.add_argument(
        "--outdir", type=str, required=True, help="Directory to save output JSON files"
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.7, help="Ratio of training data"
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.2, help="Ratio of validation data"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for splitting"
    )
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.outdir, exist_ok=True)

    # Get all seed IDs first
    all_seed_ids = set()
    for root, _, files in os.walk(args.dir):
        for file in files:
            if file.endswith("_crop.png"):
                parts = file.replace("_crop.png", "").split("_")
                roi_num = parts[1]  # e.g., "23"
                plate = parts[2]  # e.g., "plate3"
                group = parts[3]  # e.g., "ALL"
                experiment = os.path.basename(
                    os.path.dirname(root)
                )  # e.g., "2020.03.20 WP2 -N Exp7"
                seed_id = f"{roi_num}_{plate}_{group}_{experiment.replace(' ', '_')}"
                all_seed_ids.add(seed_id)

    # Split seed IDs
    train_ids, val_ids, test_ids = split_by_seed_id(
        list(all_seed_ids),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        random_state=args.seed,
    )

    # Create separate COCO JSONs for each split
    create_coco_json(
        args.dir, os.path.join(args.outdir, "train_coco.json"), set(train_ids)
    )
    create_coco_json(args.dir, os.path.join(args.outdir, "val_coco.json"), set(val_ids))
    create_coco_json(
        args.dir, os.path.join(args.outdir, "test_coco.json"), set(test_ids)
    )

    print(f"Created COCO JSON files in {args.outdir}")
    print(f"Train seeds: {len(train_ids)}")
    print(f"Val seeds: {len(val_ids)}")
    print(f"Test seeds: {len(test_ids)}")
