import argparse
import csv
import numpy as np
import sys
import os
import pickle
import nibabel as nib
import concurrent.futures
from multiprocessing import cpu_count

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from glob import glob
from natsort import natsorted
from PIL import Image
from tqdm import tqdm
from vessel_graph_generation.tree2img import rasterize_forest, voxelize_forest

"""
You can use this script to visualize previously generated vessel graphs in your requested resolution.
You can generate the rendered image, the label map and the 3D volume.
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--source_dir', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--resolution', help="The number of pixels for every dimension of the final image or volume seperated by comma.", type=str, default='1216,1216,16')
    parser.add_argument('--save_2d', action='store_true', help="Save 2d image.")
    parser.add_argument('--no_save_2d', action="store_false", dest="save_2d", help="Do not save 2d image.")
    parser.add_argument('--save_3d', help="Save 3d volume.", action="store_true")
    parser.add_argument('--save_3d_as', choices=[".nii.gz", ".npy"], default=".nii.gz", help="Save 3d volume as nifti file or numpy array. Default is nifti")
    parser.add_argument('--mip_axis', help="Axis along which to take the Mean intesity projection. Default is the z-dimension.", type=int, default=2)
    parser.add_argument('--binarize', help="Create a label map by binarizing the image", action="store_true")
    parser.add_argument('--num_samples', type=int, default=9999999, help="Number of samples to visualize. Default is all.")
    parser.add_argument('--max_dropout_prob', type=float, default=0, help="Maximum dropout probability for vessel segments to diversify training data. Default is no exclusion of vessels.")
    parser.add_argument('--ignore_z', action="store_true", default=False, help="Ignore depth dimension for 3D rendering")
    parser.add_argument('--threads', help="Number of parallel threads. By default all available threads but one are used.", type=int, default=-1)
    parser.set_defaults(save_2d=True)
    args = parser.parse_args()

    resolution = np.array([int(d) for d in args.resolution.split(',')])
    assert not args.save_3d or len(resolution)==3, "If you want to generate the 3d volume, you need to specify the resolution of all three dimensions."
    print(f"Your provided volume dimensions are {resolution}. Make sure to use the same size relations as used for generating the graph.")
    assert os.path.isdir(args.source_dir), f"The provided source directory {args.source_dir} does not exist."
    assert args.mip_axis in [0,1,2], "The axis must be '0' (x), '1' (y) or '2' (z)."
    assert args.save_3d or args.save_2d, "You must either activate saving the 2D image or the 3D volume."

    os.makedirs(args.out_dir, exist_ok=True)

    if args.save_2d:
        if len(resolution)==3:
            img_res = [*resolution]
            del img_res[args.mip_axis]
        else:
            img_res = resolution

    csv_files = natsorted(glob(os.path.join(args.source_dir, "**", "*.csv"), recursive=True))
    csv_files = csv_files[:args.num_samples]
    assert len(csv_files)>0, f"Your provided source directory {args.source_dir} does not contain any csv files."
    
    def render_graph(file_path: str):
        name = file_path.split("/")[-1].removesuffix(".csv")
        f: list[dict] = list()
        with open(file_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                f.append(row)

        if args.save_3d:
            vol,black_dict = voxelize_forest(f, resolution, max_dropout_prob=args.max_dropout_prob, ignore_z=args.ignore_z)
            if args.binarize:
                name+="_3d_label"
                vol[vol<0.1]=0
                vol[vol>=0.1]=1
                vol = vol.astype(np.bool_)
            else:
                name+="_3d"
            if args.save_3d_as == ".nii.gz":
                nifti = nib.Nifti1Image(vol, np.eye(4))
                nib.save(nifti, os.path.join(args.out_dir, name+".nii.gz"))
            else:
                np.save(os.path.join(args.out_dir, name+".npy"), vol.astype(np.bool_))
            if args.max_dropout_prob>0:
                with open(os.path.join(args.out_dir, name+"_blackdict.pkl"), 'wb') as file:
                    pickle.dump(black_dict, file)
        if args.save_2d:
            img, _ = rasterize_forest(f, img_res, args.mip_axis)
            if args.binarize:
                img[img<0.1]=0
                Image.fromarray(img.astype(np.uint8)).convert("1").save(os.path.join(args.out_dir, name+"_label.png"))
            else:
                Image.fromarray(img.astype(np.uint8)).save(os.path.join(args.out_dir, name+".png"))
            if args.max_dropout_prob>0:
                with open(os.path.join(args.out_dir, name+"_blackdict.pkl"), 'wb') as file:
                    pickle.dump(black_dict, file)

    if args.threads == -1:
        # If no argument is provided, use all available threads but one
        cpus = cpu_count()
        threads = min(cpus-1,len(csv_files)) if cpus>1 else 1
    else:
        threads=args.threads

    if threads>1:
        # Multi processing
        with tqdm(total=len(csv_files), desc="Rendering vessel graphs...") as pbar:
            with concurrent.futures.ProcessPoolExecutor(max_workers=threads) as executor:
                future_dict = {executor.submit(render_graph, csv_files[i]): i for i in range(len(csv_files))}
                for future in concurrent.futures.as_completed(future_dict):
                    i = future_dict[future]
                    pbar.update(1)
    else:
        # Single processing
        for csv_path in tqdm(csv_files, desc="Rendering vessel graphs..."):
            render_graph(csv_path)
