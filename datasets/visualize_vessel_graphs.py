import argparse
import csv
import numpy as np
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from glob import glob
from natsort import natsorted
from PIL import Image
from tqdm import tqdm
from vessel_graph_generation.tree2img import rasterize_forest

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--source_dir', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--factor', help="Factor that determines the final image resolution. In multiples of 76.", type=int, default=16)
    parser.add_argument('--binarize', help="Create a label map by binarizing the image", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    csv_files = natsorted(glob(os.path.join(args.source_dir, "**", "*.csv"), recursive=True))
    for file_path in tqdm(csv_files, desc="Visualizing images..."):
        name = file_path.split("/")[-1].removesuffix(".csv")
        f: list[dict] = list()
        with open(file_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                f.append(row)
        img, _ = rasterize_forest(f, args.factor, [],)
        if args.binarize:
            img[img<0.1]=0
            Image.fromarray(img.astype(np.uint8)).convert("1").save(os.path.join(args.out_dir, name+".png"))
        else:
            Image.fromarray(img.astype(np.uint8)).save(os.path.join(args.out_dir, name+".png"))
