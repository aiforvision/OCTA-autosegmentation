#!/bin/bash
echo "[Info] Mode: $1"
mode=$1
shift

if [ "$mode" = "segmentation" ]
then 
    python /home/OCTA-seg/test.py --config_file /home/OCTA-seg/docker/trained_models/ves_seg-S-GAN/config.yml --epoch 30 "$@" && \
    chmod -R 777 /var/segmented
elif [ "$mode" = "generation" ]
then
    num_samples=$1
    shift
    python /home/OCTA-seg/generate_vessel_graph.py --config_file /home/OCTA-seg/docker/vessel_graph_gen_docker_config.yml --num_samples $num_samples \
    && python /home/OCTA-seg/test.py --config_file /home/OCTA-seg/docker/trained_models/GAN/config.yml --epoch 150 \
    && python /home/OCTA-seg/visualize_vessel_graphs.py --source_dir /var/generation/vessel_graphs --out_dir /var/generation/labels --resolution "1216,1216,16" --binarize "$@" && \
    chmod -R 777 /var/generation
elif [ "$mode" = "transformation" ]
then
    python /home/OCTA-seg/test.py --config_file /home/OCTA-seg/docker/trained_models/GAN/config.yml --epoch 150 "$@" && \
    chmod -R 777 /var/generation/images
elif [ "$mode" = "visualization" ]
then
    python /home/OCTA-seg/visualize_vessel_graphs.py --source_dir /var/vessel_graphs --out_dir /var/labels --resolution "1216,1216,16" --binarize "$@" && \
    chmod -R 777 /var/labels
elif [ "$mode" = "3d_reconstruction" ]
then
    python /home/OCTA-seg/test.py --config_file /home/OCTA-seg/docker/trained_models/reconstruction_3d/config.yml --epoch 60 "$@" && \
    chmod -R 777 /var/reconstructed
else
    echo "Mode $mode does not exist. Choose segmentation, generation or translation."
    exit 1
fi