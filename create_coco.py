import os
import json
import cv2
import argparse


def create_coco_json(data_dir, output_json):
    # initialize COCO format structure
    coco_format = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "germinated", "supercategory": "seed"},
            {"id": 2, "name": "ungerminated", "supercategory": "seed"},
        ],
    }

    annotation_id = 1
    image_id = 1

    # traverse the directory structure
    for experiment in os.listdir(data_dir):
        experiment_path = os.path.join(data_dir, experiment)
        if not os.path.isdir(experiment_path):
            continue  # skip non-directories

        for category in ["germinated", "ungerminated"]:
            category_path = os.path.join(experiment_path, category)
            if not os.path.isdir(category_path):
                continue  # skip if category folder is missing

            category_id = 1 if category == "germinated" else 2

            for file in os.listdir(category_path):
                if not file.endswith("_crop.png"):
                    continue  # only process crop images

                # extract metadata from file name
                base_name = file.replace("_crop.png", "")
                seed_id = int(base_name.split("_")[1])  # extract id from e.g., roi_1
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

                # add image metadata
                coco_format["images"].append(
                    {
                        "id": image_id,
                        "file_name": f"{experiment}/{category}/{file}",
                        "width": width,
                        "height": height,
                        "experiment": experiment,  # include experiment name
                        "seed_id": str(seed_id)
                        + "_"
                        + base_name.split("_")[2]
                        + "_"
                        + experiment.replace(" ", "_"),
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
        "--outfile", type=str, required=True, help="Path to output JSON file"
    )
    args = parser.parse_args()

    create_coco_json(args.dir, args.outfile)
