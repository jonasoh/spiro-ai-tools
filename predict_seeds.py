import torch
import cv2
import argparse
import torchvision
import pandas as pd
from PIL import Image
from torchvision.transforms import ToTensor


def load_model(weights_path, num_classes=3):
    """loads and configures the mask r-cnn model"""
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None)
    model.roi_heads.box_predictor = (
        torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
            model.roi_heads.box_predictor.cls_score.in_features, num_classes
        )
    )
    model.roi_heads.mask_predictor = (
        torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
            model.roi_heads.mask_predictor.conv5_mask.in_channels, 256, num_classes
        )
    )
    model.load_state_dict(torch.load(weights_path, weights_only=True))
    return model.eval()


def get_seed_regions(
    tsv_path, slice_number, image_shape, crop_width=128, crop_height=256
):
    """Extract seed regions based on coordinates in TSV file"""
    df = pd.read_csv(tsv_path, sep="\t")
    slice_data = df[df["Slice"] == slice_number]

    seed_regions = []
    for _, row in slice_data.iterrows():
        x_center = row["xUP"]
        y_center = row["yUP"] + int(
            crop_height / 6
        )  # Apply y-offset as in prepare_spiro_data.py

        # Calculate crop box boundaries
        x_start = max(0, int(x_center - crop_width / 2))
        y_start = max(0, int(y_center - crop_height / 2))
        x_end = min(image_shape[1], x_start + crop_width)
        y_end = min(image_shape[0], y_start + crop_height)

        seed_regions.append(
            {
                "bbox": (x_start, y_start, x_end - x_start, y_end - y_start),
                "roi": int(row["ROI"]),
            }
        )

    return seed_regions


def process_image(image_path, model, seed_regions, confidence_threshold=0.5):
    """Process an image using predefined seed regions"""
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictions = []

    transform = ToTensor()

    for region in seed_regions:
        x, y, w, h = region["bbox"]

        # Extract and preprocess seed image
        seed_image = image_rgb[y : y + h, x : x + w]
        seed_tensor = transform(Image.fromarray(seed_image)).unsqueeze(0)

        # Get prediction
        with torch.no_grad():
            pred = model(seed_tensor)

        # Process predictions above confidence threshold
        boxes = pred[0]["boxes"]
        scores = pred[0]["scores"]
        labels = pred[0]["labels"]
        masks = pred[0]["masks"]

        high_conf_idx = scores > confidence_threshold
        if high_conf_idx.any():
            predictions.append(
                {
                    "bbox": region["bbox"],
                    "roi": region["roi"],
                    "pred_boxes": boxes[high_conf_idx].numpy(),
                    "scores": scores[high_conf_idx].numpy(),
                    "labels": labels[high_conf_idx].numpy(),
                    "masks": masks[high_conf_idx].numpy(),
                }
            )

    return image_rgb, predictions


def draw_predictions(
    image, predictions, class_names=["background", "germinated", "ungerminated"]
):
    """draw predictions on the image"""
    result = image.copy()
    colors = [(0, 255, 0), (0, 0, 255)]  # green for germinated, red for ungerminated

    for pred in predictions:
        x, y, w, h = pred["bbox"]

        # draw region rectangle
        cv2.rectangle(result, (x, y), (x + w, y + h), (255, 255, 255), 2)

        # draw classification and roi number inside box
        label_idx = pred["labels"][0] - 1  # subtract 1 to skip background class
        label_text = f"{pred['roi']}: {class_names[label_idx+1]}"

        # use smaller font size
        font_scale = 0.4
        thickness = 1

        # get text size to position properly
        (text_width, text_height), _ = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )

        # position text inside box with small padding
        text_x = x + 5
        text_y = y + text_height + 5

        # draw text with small outline for better visibility
        cv2.putText(
            result,
            label_text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),  # black outline
            thickness + 1,
        )
        cv2.putText(
            result,
            label_text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            colors[label_idx],
            thickness,
        )

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Analyze seeds in an image using trained R-CNN model"
    )
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument(
        "--coords", required=True, help="Path to TSV file with seed coordinates"
    )
    parser.add_argument(
        "--slice", type=int, required=True, help="Slice number to process"
    )
    parser.add_argument("--weights", required=True, help="Path to model weights")
    parser.add_argument("--output", required=True, help="Path to output image")
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="Confidence threshold"
    )

    args = parser.parse_args()

    # Load model
    model = load_model(args.weights)

    # Get image shape
    image = cv2.imread(args.image)
    if image is None:
        raise ValueError(f"Could not open image: {args.image}")

    # Get seed regions from TSV
    seed_regions = get_seed_regions(args.coords, args.slice, image.shape)

    # Process image
    image, predictions = process_image(args.image, model, seed_regions, args.threshold)

    # Draw predictions
    result = draw_predictions(image, predictions)

    # Save result
    result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    cv2.imwrite(args.output, result_bgr)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
