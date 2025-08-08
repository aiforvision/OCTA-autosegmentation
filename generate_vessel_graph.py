import argparse
import concurrent.futures
import csv
import os
import warnings
from multiprocessing import cpu_count

import nibabel as nib
import numpy as np
import vessel_graph_generation.tree2img as tree2img
import yaml
from rich.console import Console, Group
from rich.live import Live
from rich.progress import Progress, TimeElapsedColumn
from utils.config_overrides import apply_cli_overrides_from_unknown_args
from utils.visualizer import DynamicDisplay
from vessel_graph_generation.forest import Forest
from vessel_graph_generation.greenhouse import Greenhouse
from vessel_graph_generation.utilities import prepare_output_dir, read_config

group = Group()


def main(config):
    # Initialize greenhouse
    greenhouse = Greenhouse(config['Greenhouse'])
    # Prepare output directory
    out_dir = prepare_output_dir(config['output'])
    with open(os.path.join(out_dir, 'config.yml'), 'w') as f:
        yaml.dump(config, f)

    # Initialize forest
    arterial_forest = Forest(config['Forest'], greenhouse.d, greenhouse.r, greenhouse.simspace, nerve_center=greenhouse.nerve_center, nerve_radius=greenhouse.nerve_radius)
    venous_forest = Forest(config['Forest'], greenhouse.d, greenhouse.r, greenhouse.simspace, arterial=False, nerve_center=greenhouse.nerve_center, nerve_radius=greenhouse.nerve_radius)#, template_forest=arterial_forest)
    
    greenhouse.set_forests(arterial_forest, venous_forest)

    # Grow vessel network
    greenhouse.develop_forest()
    if config["output"]["save_stats"]:
        greenhouse.save_stats(out_dir)

    volume_dimension = [int(d) for d in greenhouse.simspace.shape*config['output']['image_scale_factor']]

    art_edges = [{
        "node1": current_node.position,
        "node2": current_node.get_proximal_node().position,
        "radius": current_node.radius
    } for tree in arterial_forest.get_trees() for current_node in tree.get_tree_iterator(exclude_root=True, only_active=False)]

    if venous_forest is not None:
        ven_edges = [{
            "node1": current_node.position,
            "node2": current_node.get_proximal_node().position,
            "radius": current_node.radius
        } for tree in venous_forest.get_trees() for current_node in tree.get_tree_iterator(exclude_root=True, only_active=False)]

    # Save vessel graph as csv file
    if config['output']['save_trees']:
        name = out_dir.split("/")[-1]
        filepath = os.path.join(out_dir, name+'.csv')
        with open(filepath, 'w+') as file:
            writer = csv.writer(file)
            writer.writerow(["node1", "node2", "radius"])
            for row in art_edges+ven_edges:
                writer.writerow([row["node1"], row["node2"], row["radius"]])

    radius_list=[]
    if config["output"].get("save_3D_volumes"):
        art_mat, _ = tree2img.voxelize_forest(art_edges, volume_dimension, radius_list)
        ven_mat, _ = tree2img.voxelize_forest(ven_edges, volume_dimension, radius_list)
        art_ven_mat_gray = np.maximum(art_mat, ven_mat).astype(np.uint8)
        if config["output"]["save_3D_volumes"] == "npy":
            np.save(f'{out_dir}/art_ven_img_gray.npy', art_ven_mat_gray)
        elif config["output"]["save_3D_volumes"] == "nifti":
            nifti = nib.Nifti1Image(art_ven_mat_gray, np.eye(4))
            nib.save(nifti, f"{out_dir}/art_ven_img_gray.nii.gz")
    
    if config["output"]["save_2D_image"]:
        radius_list=[]
        image_res = [*volume_dimension]
        del image_res[config["output"]["proj_axis"]]
        art_mat,_ = tree2img.rasterize_forest(art_edges, image_res, MIP_axis=config["output"]["proj_axis"], radius_list=radius_list)
        ven_mat,_ = tree2img.rasterize_forest(ven_edges, image_res, MIP_axis=config["output"]["proj_axis"], radius_list=radius_list)
        art_ven_mat_gray = np.maximum(art_mat, ven_mat).astype(np.uint8)
        tree2img.save_2d_img(art_ven_mat_gray, out_dir, "art_ven_img_gray")

    if config["output"]["save_stats"]:
        tree2img.plot_vessel_radii(out_dir, radius_list)

if __name__ == '__main__':
    # Parse input arguments
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config_file', type=str, required=True)
    parser.add_argument('--num_samples', type=int, default=1)
    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--threads', help="Number of parallel threads. By default all available threads but one are used.", type=int, default=-1)
    args, _unknown_args = parser.parse_known_args()

    if args.debug:
        warnings.filterwarnings('error')

    # Read config file
    assert os.path.isfile(args.config_file), f"Error: Your provided config path {args.config_file} does not exist!"
    config = read_config(args.config_file)

    # Apply CLI overrides before using config
    apply_cli_overrides_from_unknown_args(config, _unknown_args)

    assert config['output'].get('save_3D_volumes') in [None, 'npy', 'nifti'], f"Your provided option {config['output'].get('save_3D_volumes')} for 'save_3D_volumes' does not exist. Choose one of 'null', 'npy' or 'nifti'."

    if args.threads == -1:
        # If no argument is provided, use all available threads but one
        cpus = cpu_count()
        threads = min(cpus-1 if cpus>1 else 1,args.num_samples)
    else:
        threads=args.threads

    with Live(group, console=Console(force_terminal=True), refresh_per_second=10):
        progress = Progress(*Progress.get_default_columns(), TimeElapsedColumn())
        progress.add_task(f"Generating {args.num_samples} vessel graphs:", total=args.num_samples)
        with DynamicDisplay(group, progress):
            if threads>1:
                # Multi processing
                with concurrent.futures.ProcessPoolExecutor(max_workers=threads) as executor:
                    future_dict = {executor.submit(main, config): i for i in range(args.num_samples)}
                    for future in concurrent.futures.as_completed(future_dict):
                        i = future_dict[future]
                        progress.advance(task_id=0)
            else:
                # Single processing
                for i in range(args.num_samples):
                    main(config)
                    progress.advance(task_id=0)
