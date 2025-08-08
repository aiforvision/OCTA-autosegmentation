import argparse
import csv
import glob
import os
import warnings

import numpy as np
from natsort import natsorted
from PIL import Image
from tqdm import tqdm

warnings.filterwarnings('error')

"""
This script is used to crop the ROI from the images. It is assumed that the ROI is located in the center of the image.
If the image is larger than the specified ROI size, the script will try to find the ROI by looking for the
largest difference in pixel values between the ROI and the surrounding area. The script will then crop the image to the roi_size
around the found ROI. If the image is smaller than the roi_size, the script will pad the image with zeros to reach the target size.
"""


def calculate_roi_coordinates(img, image_size, roi_size):
    """
    Calculate ROI coordinates by analyzing pixel differences in the image.
    
    Args:
        img: Input image as numpy array
        image_size: Size of the image (minimum dimension)
        roi_size: Target ROI size
    
    Returns:
        tuple: (xs, ys) coordinates for ROI cropping
    """
    # Calculate differences in the top-left region
    third = image_size // 3
    
    # X-direction differences
    diff_xx = (img[:third, third:third*2] - img[1:third+1, third:third*2]).sum(axis=1)
    diff_xy = abs(img[:third, third:third*2] - img[:third, third+1:third*2+1]).sum(axis=1)
    xxs = np.argmax(diff_xx) + 1
    xys = np.argmin(diff_xy[:-1] - diff_xy[1:]) + 1

    # Y-direction differences
    diff_yx = abs(img[third:third*2, :third] - img[third+1:third*2+1, :third]).sum(axis=0)
    diff_yy = (img[third:third*2, :third] - img[third:third*2, 1:third+1]).sum(axis=0)
    yxs = np.argmin(diff_yx[:-1] - diff_yx[1:]) + 1
    yys = np.argmax(diff_yy) + 1

    # Calculate reverse coordinates using flipped image
    img_flip = np.flip(np.flip(img, axis=0), axis=1)
    diff_xx_reverse = (img_flip[:third, third:third*2] - img_flip[1:third+1, third:third*2]).sum(axis=1)
    diff_xy_reverse = abs(img_flip[:third, third:third*2] - img_flip[:third, third+1:third*2+1]).sum(axis=1)
    xxs_reverse = image_size - (np.argmax(diff_xx_reverse) + 1) - roi_size
    xys_reverse = image_size - (np.argmin(diff_xy_reverse[:-1] - diff_xy_reverse[1:]) + 1) - roi_size
    
    diff_yy_reverse = (img_flip[third:third*2, :third] - img_flip[third:third*2, 1:third+1]).sum(axis=0)
    diff_yx_reverse = abs(img_flip[third:third*2, :third] - img_flip[third+1:third*2+1, :third]).sum(axis=0)
    yxs_reverse = image_size - (np.argmin(diff_yx_reverse[:-1] - diff_yx_reverse[1:]) + 1) - roi_size
    yys_reverse = image_size - (np.argmax(diff_yy_reverse) + 1) - roi_size

    # Choose the most common coordinates
    xs_list = [xxs, xys, xxs_reverse, xys_reverse]
    xs = max(set(xs_list), key=xs_list.count)
    ys_list = [yxs, yys, yxs_reverse, yys_reverse]
    ys = max(set(ys_list), key=ys_list.count)

    return xs, ys


def create_problematic_entry(path, save_path=None, shape=None, xs=None, ys=None):
    """Create a dictionary entry for problematic images."""
    return {
        "path": path,
        "save_path": save_path,
        "shape": shape,
        "xs": xs,
        "ys": ys
    }


def is_problematic_crop(img_cropped, xs, ys, roi_size, image_size, problem_threshold):
    """Check if the cropped image is problematic based on size and position."""
    wrong_shape = img_cropped.shape[0] != roi_size or img_cropped.shape[1] != roi_size
    too_close_to_edge = xs < problem_threshold * image_size or ys > (1 - problem_threshold) * image_size
    return wrong_shape or too_close_to_edge


if __name__ == "__main__":
    description = (
        "This script crops the ROI from the images in the input directory and saves them to the output directory. "
        "It assumes that the ROI is located in the center of the image. If the image is larger than the specified ROI size, "
        "the script will try to find the ROI by looking for the largest difference in pixel values between the ROI and the surrounding area. "
        "The script will then crop the image to the roi_size around the found ROI. If the image is smaller than the roi_size, "
        "the script will pad the image with zeros to reach the target size. "
        "The script will also save a CSV file with the problematic images that could not be cropped correctly. "
        "The problematic images are those that do not have the expected shape after cropping or are cropped too close to the edges of the image."
    )
    
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--roi_size', type=int, default=512)
    parser.add_argument('--problem_threshold', type=float, default=0.15)

    args = parser.parse_args()
    roi_size = args.roi_size

    data_files = natsorted(glob.glob(f'{args.input_dir}/**/*.png', recursive=True))
    assert len(data_files) > 0, f"No input files found for path {args.input_dir}"

    # Check if input directory has subfolders with images
    input_dir_normalized = os.path.normpath(args.input_dir)
    has_subfolders = any(
        os.path.dirname(os.path.normpath(path)) != input_dir_normalized 
        for path in data_files
    )

    problematic = []
    
    for path in tqdm(data_files):
        if not os.path.isfile(path):
            continue
            
        name = path.split("/")[-1].replace(".PNG", ".png")
        
        # Only preserve directory structure if input has subfolders
        if has_subfolders:
            # Get relative path from input directory to preserve subfolder structure
            rel_path = os.path.relpath(os.path.dirname(path), args.input_dir)
            if rel_path == ".":  # File is directly in input directory
                cohort = ""
            else:
                cohort = rel_path
        else:
            cohort = ""
        
        try:
            img = np.array(Image.open(path).convert("L")).astype(np.float32)
        except OSError:
            problematic.append(create_problematic_entry(path))
            continue
        
        # Get image dimensions dynamically
        image_height, image_width = img.shape
        image_size = min(image_height, image_width)  # Use minimum dimension to ensure we stay within bounds
        
        if img.shape[0] > roi_size + 1 and img.shape[1] > roi_size + 1:
            xs, ys = calculate_roi_coordinates(img, image_size, roi_size)
            img_cropped = img[xs:xs+roi_size, ys:ys+roi_size].astype(np.uint8)
        else:
            xs, ys = 0, 0  # Initialize for cases where image is smaller than roi_size
            img_cropped = img[:roi_size, :roi_size].astype(np.uint8)

        # Create output directory structure if needed
        if cohort:
            output_cohort_dir = os.path.join(args.output_dir, cohort)
            if not os.path.exists(output_cohort_dir):
                os.makedirs(output_cohort_dir)
            save_path = os.path.join(output_cohort_dir, name)
        else:
            # Save directly to output directory
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)
            save_path = os.path.join(args.output_dir, name)
        
        # Check if the crop is problematic
        if is_problematic_crop(img_cropped, xs, ys, roi_size, image_size, args.problem_threshold):
            if (img_cropped.shape[0] != roi_size or img_cropped.shape[1] != roi_size):
                problematic.append(create_problematic_entry(
                    path, save_path, (img_cropped.shape[0], img_cropped.shape[1]), xs, ys
                ))
        
        # Ensure the output image has the correct size by padding with zeros if necessary
        img_cropped_final = np.zeros((roi_size, roi_size), dtype=np.uint8)
        img_cropped_final[:img_cropped.shape[0], :img_cropped.shape[1]] = img_cropped[:roi_size, :roi_size]

        Image.fromarray(img_cropped_final).save(save_path)

    # Save problematic images to CSV
    with open(f"{args.output_dir}/problematic.csv", 'w+') as csvfile:
        writer = csv.writer(csvfile)
        if len(problematic) > 0:
            writer.writerow(list(problematic[0].keys()))
            for entry in problematic:
                writer.writerow(entry.values())
        else:
            writer.writerow(["ALL CLEAR"])
    