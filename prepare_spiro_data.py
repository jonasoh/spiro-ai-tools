import os
import cv2
import sys
import pims
import glob
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from parallel_save import parallel_save_images

parser = argparse.ArgumentParser(
    description="Process and crop SPIRO images (run root growth macro in debug mode first!)"
)
parser.add_argument(
    "--indir",
    "-i",
    required=True,
    help="Input directory containing image files. This is the base directory containing your SPIRO experiment",
)
parser.add_argument(
    "--outdir",
    "-o",
    required=True,
    help="Output directory to save cropped images. Will be created if missing.",
)
parser.add_argument(
    "--width", type=int, default=128, help="Width of the crop box (default: 150)."
)
parser.add_argument(
    "--height",
    type=int,
    default=256,
    help="Height of the crop box (default: 300).",
)
parser.add_argument(
    "--nocrop",
    action="store_true",
    help="Whether to crop out individual seeds (default: True).",
)
parser.add_argument(
    "--nomask",
    action="store_true",
    help="If set, only process images and skip mask processing.",
)
parser.add_argument(
    "--verbose", action="store_true", help="Print verbose output to stderr."
)

# parse arguments
args = parser.parse_args()
w, h = args.width, args.height
indir = os.path.abspath(args.indir)
outdir = os.path.join(os.path.abspath(args.outdir), os.path.basename(indir))


def find_crop_position(orig, cropped_image):
    """finds the x,y top-left corner of the crop in the original picture"""
    return cv2.minMaxLoc(
        cv2.matchTemplate(
            cv2.imread(orig, cv2.IMREAD_GRAYSCALE), cropped_image, cv2.TM_CCOEFF_NORMED
        )
    )[3]


def keep_largest_object(binary_image):
    """keeps only the largest connected object in a binary image"""
    cleaned_image = np.zeros_like(binary_image)
    contours, _ = cv2.findContours(
        binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if contours:
        cv2.drawContours(
            cleaned_image,
            [max(contours, key=cv2.contourArea)],
            -1,
            255,
            thickness=cv2.FILLED,
        )
    return cleaned_image


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
    # ensure the point (x, y) is within bounds
    if not (0 <= x < binary_image.shape[1] and 0 <= y < binary_image.shape[0]):
        raise ValueError(f"Point ({x}, {y}) is outside the image bounds.")

    # check if the point belongs to an object
    if binary_image[y, x] == 0:
        raise ValueError(f"Point ({x}, {y}) does not belong to any object.")

    # create a mask for floodFill
    mask = np.zeros(
        (binary_image.shape[0] + 2, binary_image.shape[1] + 2), dtype=np.uint8
    )

    # floodFill to get the connected component
    flood_filled_image = binary_image.copy()
    _, _, _, rect = cv2.floodFill(
        flood_filled_image, mask, seedPoint=(x, y), newVal=128
    )

    # extract the connected component
    connected_component = (flood_filled_image == 128).astype(np.uint8) * 255

    # filter by size if min_size > 0
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
            return np.zeros_like(binary_image)  # no object meets size criteria

    return connected_component


def get_objects_from_coordinates(binary_image, coordinates):
    """
    Keeps objects connected to specific coordinates in a binary image.

    Parameters:
        binary_image (numpy.ndarray): Binary image (grayscale), where objects are white (255) and background is black (0).
        coordinates (list of tuples): List of (x, y) coordinates.

    Returns:
        numpy.ndarray: Binary mask containing only the objects connected to the specified coordinates.
    """
    # create an empty mask to store the final result
    final_mask = np.zeros_like(binary_image, dtype=np.uint8)

    # make a copy of the binary image to track which regions have been processed
    processed_image = binary_image.copy()

    for x, y in coordinates:
        # ensure the point is within bounds
        if not (
            0 <= x < processed_image.shape[1] and 0 <= y < processed_image.shape[0]
        ):
            if args.verbose:
                print(
                    f"Skipping ({x}, {y}): point is outside image bounds.",
                    file=sys.stderr,
                )
            continue

        # check if the point belongs to an object
        if processed_image[y, x] == 0:
            if args.verbose:
                print(
                    f"Skipping ({x}, {y}): point does not belong to any object.",
                    file=sys.stderr,
                )
            continue

        # create a mask for floodFill
        mask = np.zeros(
            (processed_image.shape[0] + 2, processed_image.shape[1] + 2), dtype=np.uint8
        )

        # floodFill to extract the connected component
        flood_filled_image = processed_image.copy()
        cv2.floodFill(flood_filled_image, mask, seedPoint=(x, y), newVal=128)

        # add the connected object to the final mask
        final_mask[flood_filled_image == 128] = 255

        # mark the object as processed to avoid overlaps
        processed_image[flood_filled_image == 128] = 0

    return final_mask


def save_nocrop_image(slice_number, orig_files, firstslice, masks, df):
    # open the corresponding image
    # we assume that imagej analysis was started from the first slice
    image_path = orig_files[slice_number - 1]
    image = cv2.imread(os.path.join(indir, d["plate"], image_path))

    # crop the image to the same dimensions as the first slice
    cropped_image = image[
        TOPLEFT[1] : TOPLEFT[1] + firstslice[0].shape[0],
        TOPLEFT[0] : TOPLEFT[0] + firstslice[0].shape[1],
    ]

    mask = get_objects_from_coordinates(
        masks[slice_number - 1],
        zip(
            df.loc[df["Slice"] == slice_number, "xUP"].astype(int),
            df.loc[df["Slice"] == slice_number, "yUP"].astype(int),
        ),
    )

    # save the cropped whole image and corresponding mask
    output_path = os.path.join(outdir, f"{d['code']}_{slice_number}_crop.webp")
    cv2.imwrite(output_path, cropped_image)
    cv2.imwrite(
        output_path.replace("_crop.webp", "_mask.webp"),
        mask,
    )


def format_slice_number(slice_num, total_slices):
    """Format slice number with leading zeros based on total number of slices"""
    width = len(str(total_slices))
    return str(slice_num).zfill(width)


def process_cropped_image(image, x_center, y_center, w, h, TOPLEFT):
    """Extract a cropped region from the image."""
    x_start = max(0, int(x_center - w / 2))
    y_start = max(0, int(y_center - h / 2))
    x_end = min(image.shape[1], x_start + w)
    y_end = min(image.shape[0], y_start + h)

    return image[
        TOPLEFT[1] + y_start : TOPLEFT[1] + y_end,
        TOPLEFT[0] + x_start : TOPLEFT[0] + x_end,
    ], (x_start, y_start, x_end, y_end)


def process_uncropped_image(image, TOPLEFT, firstslice):
    """Process the full image without cropping."""
    return image[
        TOPLEFT[1] : TOPLEFT[1] + firstslice[0].shape[0],
        TOPLEFT[0] : TOPLEFT[0] + firstslice[0].shape[1],
    ]


def prepare_save_data(image, mask=None):
    """Prepare image and optional mask for saving."""
    saves = [Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))]
    if mask is not None:
        saves.append(Image.fromarray(mask))
    return saves


def process_images(d, df, orig_files, masks, firstslice, TOPLEFT, args, w, h):
    """Main image processing function."""
    max_slice = df["Slice"].max()
    images_to_save = []
    paths_to_save = []
    stats = {"success": 0, "failed": 0}

    if args.nocrop:
        # Process full images
        with tqdm(total=df["Slice"].nunique(), desc="Processing images") as pbar:
            for slice_number, group in df.groupby("Slice"):
                try:
                    image = cv2.imread(
                        os.path.join(indir, d["plate"], orig_files[slice_number - 1])
                    )
                    cropped_image = process_uncropped_image(image, TOPLEFT, firstslice)

                    if not args.nomask:
                        try:
                            mask = get_objects_from_coordinates(
                                masks[slice_number - 1],
                                zip(
                                    df.loc[df["Slice"] == slice_number, "xUP"].astype(
                                        int
                                    ),
                                    df.loc[df["Slice"] == slice_number, "yUP"].astype(
                                        int
                                    ),
                                ),
                            )
                            base_path = os.path.join(
                                outdir,
                                f"{d['code']}_{format_slice_number(slice_number, max_slice)}_crop.webp",
                            )
                            images_to_save.extend(
                                prepare_save_data(cropped_image, mask)
                            )
                            paths_to_save.extend(
                                [
                                    base_path,
                                    base_path.replace("_crop.webp", "_mask.webp"),
                                ]
                            )
                            stats["success"] += 1
                        except Exception as e:
                            stats["failed"] += 1
                    else:
                        base_path = os.path.join(
                            outdir,
                            f"{d['code']}_{format_slice_number(slice_number, max_slice)}_crop.webp",
                        )
                        images_to_save.extend(prepare_save_data(cropped_image))
                        paths_to_save.append(base_path)
                        stats["success"] += 1
                except Exception as e:
                    stats["failed"] += 1
                pbar.update(1)
    else:
        # Process individual ROIs
        with tqdm(total=len(df), desc="Processing ROIs") as pbar:
            for slice_number, group in df.groupby("Slice"):
                image = cv2.imread(
                    os.path.join(indir, d["plate"], orig_files[slice_number - 1])
                )

                for _, row in group.iterrows():
                    roi = int(row["ROI"])
                    x_center = row["xUP"]
                    y_center = row["yUP"] + int(h / 6)

                    try:
                        cropped_image, (x_start, y_start, x_end, y_end) = (
                            process_cropped_image(
                                image, x_center, y_center, w, h, TOPLEFT
                            )
                        )

                        if not args.nomask:
                            try:
                                cropped_mask = masks[slice_number - 1][
                                    y_start:y_end, x_start:x_end
                                ]
                                cropped_mask = get_connected_object(
                                    cropped_mask, 64, 87
                                )
                                base_path = os.path.join(
                                    outdir,
                                    f"roi_{roi}_{d['code']}_{format_slice_number(slice_number, max_slice)}_crop.webp",
                                )
                                images_to_save.extend(
                                    prepare_save_data(cropped_image, cropped_mask)
                                )
                                paths_to_save.extend(
                                    [
                                        base_path,
                                        base_path.replace("_crop.webp", "_mask.webp"),
                                    ]
                                )
                                stats["success"] += 1
                            except Exception as e:
                                stats["failed"] += 1
                        else:
                            base_path = os.path.join(
                                outdir,
                                f"roi_{roi}_{d['code']}_{format_slice_number(slice_number, max_slice)}_crop.webp",
                            )
                            images_to_save.extend(prepare_save_data(cropped_image))
                            paths_to_save.append(base_path)
                            stats["success"] += 1

                    except Exception as e:
                        stats["failed"] += 1

                    pbar.update(1)

    return images_to_save, paths_to_save, stats


if __name__ == "__main__":
    # create directory list directly
    dirs = [
        {
            "code": f"{os.path.basename(os.path.dirname(d))}_{os.path.basename(d)}",
            "group": os.path.basename(d),
            "plate": os.path.basename(os.path.dirname(d)),
            "dir": d,
        }
        for d in glob.glob(os.path.join(indir, "Results/Root Growth/plate[1-4]/*"))
    ]

    total_stats = {"success": 0, "failed": 0}

    for d in dirs:
        print(f'Processing {d["code"]}...')
        firstslice = pims.open(os.path.join(d["dir"], "firstslice.tif"))
        masks = pims.open(os.path.join(d["dir"], d["group"] + " masked.tif"))
        orig_files = os.listdir(os.path.join(indir, d["plate"]))
        orig_files.sort()
        TOPLEFT = find_crop_position(
            os.path.join(indir, d["plate"], orig_files[0]), firstslice[0]
        )

        df = pd.read_csv(
            os.path.join(d["dir"], d["group"] + " rootstartcoordinates.tsv"), sep="\t"
        )

        images_to_save, paths_to_save, stats = process_images(
            d, df, orig_files, masks, firstslice, TOPLEFT, args, w, h
        )

        save_params = {"format": "WebP", "lossless": True}
        os.makedirs(outdir, exist_ok=True)
        parallel_save_images(images_to_save, paths_to_save, save_params)

        total_stats["success"] += stats["success"]
        total_stats["failed"] += stats["failed"]
        print(
            f'Finished {d["code"]}: {stats["success"]} successful, {stats["failed"]} failed',
            file=sys.stderr,
        )

    print(
        f'\nTotal processing results: {total_stats["success"]} successful, {total_stats["failed"]} failed',
        file=sys.stderr,
    )
