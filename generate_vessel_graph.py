import argparse
import csv
from vessel_graph_generation.forest import Forest
from vessel_graph_generation.greenhouse import Greenhouse
from vessel_graph_generation.utilities import prepare_output_dir, read_config
import vessel_graph_generation.tree2img as tree2img
import numpy as np
import os
from tqdm import tqdm
import yaml


def main(config):
    # Initialize greenhouse
    greenhouse = Greenhouse(config['Greenhouse'])
    # Prepare output directory
    out_dir = prepare_output_dir(config['output'])
    with open(os.path.join(out_dir, 'config.yml'), 'w') as f:
        yaml.dump(config, f)

    # Initialize forest
    arterial_forest = Forest(config['Forest'], greenhouse.simspace)
    venous_forest = Forest(config['Forest'], greenhouse.simspace, arterial=False)#, template_forest=arterial_forest)
    
    greenhouse.set_forests(arterial_forest, venous_forest)

    greenhouse.develop_forest()
    if config["output"]["save_stats"]:
        greenhouse.save_stats()

    image_scale_factor = config['output']['image_scale_factor']
    radius_list=[]

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

    if config['output']['save_trees']:
        name = out_dir.split("/")[-1]
        filepath = os.path.join(out_dir, name+'.csv')
        with open(filepath, 'w+') as file:
            writer = csv.writer(file)
            writer.writerow(["node1", "node2", "radius"])
            for row in art_edges+ven_edges:
                writer.writerow([row["node1"], row["node2"], row["radius"]])

    if config["output"]["save_3D_volumes"]:
        art_mat = tree2img.voxelize_forest(art_edges, image_scale_factor, radius_list, min_radius=0)
        if config["output"]["save_art"]:
            np.save(f'{out_dir}/art_img.npy', art_mat)
        
            ven_mat = tree2img.voxelize_forest(ven_edges, image_scale_factor, radius_list, min_radius=0)
            if config["output"]["save_ven"]:
                np.save(f'{out_dir}/ven_img.npy', ven_mat)
        else:
            ven_mat = 0

        if config["output"]["save_art_ven_gray"]:
            art_ven_mat_gray = np.maximum(art_mat, ven_mat).astype(np.uint8)
            np.save(f'{out_dir}/art_ven_img_gray.npy', art_ven_mat_gray)

        if config["output"]["save_art_ven_rgb"]:
            art_mat = np.tile(art_mat[:,:,:,np.newaxis], 3).astype(np.float64)
            red = np.tile(np.array([[[[1,0,0]]]]), (*art_mat.shape[:-1], 1))
            art_mat = art_mat * red

            ven_mat = np.tile(ven_mat[:,:,:,np.newaxis], 3).astype(np.float64)
            blue = np.tile(np.array([[[[0,0,1]]]]), (*ven_mat.shape[:-1], 1))
            ven_mat = ven_mat * blue

            art_ven_mat = np.maximum(art_mat, ven_mat)
            art_ven_mat = np.clip(art_ven_mat, 0, 255).astype(np.uint8)
            np.save(f'{out_dir}/art_ven_img_rgb.npy', art_ven_mat)

    else:
        art_mat = tree2img.rasterize_forest(art_edges, image_scale_factor, radius_list, min_radius=0)[0]
        if config["output"]["save_art"]:
            tree2img.save_2d_img(art_mat, out_dir, "art_img")

        if venous_forest is not None:
            ven_mat = tree2img.rasterize_forest(ven_edges, image_scale_factor, radius_list, min_radius=0)[0]
            if config["output"]["save_ven"]:
                tree2img.save_2d_img(ven_mat, out_dir, "ven_img")
        else:
            ven_mat = 0

        if config["output"]["save_art_ven_gray"]:
            art_ven_mat_gray = np.maximum(art_mat, ven_mat).astype(np.uint8)
            tree2img.save_2d_img(art_ven_mat_gray, out_dir, "art_ven_img_gray")
        
        if config["output"]["save_art_ven_rgb"]:
            art_mat = np.tile(art_mat[:,:,np.newaxis], 3).astype(np.float64)
            red = np.tile(np.array([[[[1,0,0]]]]), (*art_mat.shape[:-1], 1))
            art_mat = art_mat * red

            ven_mat = np.tile(ven_mat[:,:,np.newaxis], 3).astype(np.float64)
            blue = np.tile(np.array([[[[0,0,1]]]]), (*ven_mat.shape[:-1], 1))
            ven_mat = ven_mat * blue

            art_ven_mat = np.maximum(art_mat, ven_mat)
            art_ven_mat = np.clip(art_ven_mat, 0, 255).astype(np.uint8)
            tree2img.save_2d_img(ven_mat, out_dir, "art_ven_rgb")

    if config["output"]["save_stats"]:
        tree2img.plot_vessel_radii(out_dir, radius_list)

if __name__ == '__main__':
    # Parse input arguments
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config_file', type=str, required=True)
    parser.add_argument('--num_samples', type=int, default=1)
    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--threads', help="Number of parallel threads. By default all available threads but one are used.", type=int, default=-1)
    args = parser.parse_args()

    if args.debug:
        import warnings
        warnings.filterwarnings('error')

    assert os.path.isfile(args.config_file), f"Your provided config path {args.config_file} does not exist!"
    config = read_config(args.config_file)

    from multiprocessing import cpu_count
    # Read config file

    if args.threads == -1:
        cpus = cpu_count()
        threads = min(cpus-1 if cpus>1 else 1,args.num_samples)
    else:
        threads=args.threads
    if threads>1:
        import concurrent.futures
        with tqdm(total=args.num_samples, desc="Generating vessel graph...") as pbar:
            with concurrent.futures.ProcessPoolExecutor(max_workers=threads) as executor:
                future_dict = {executor.submit(main, config): i for i in range(args.num_samples)}
                for future in concurrent.futures.as_completed(future_dict):
                    i = future_dict[future]
                    pbar.update(1)
    else:
        for i in tqdm(range(args.num_samples), desc="Generating vessel graph..."):
            main(args.config_file)
