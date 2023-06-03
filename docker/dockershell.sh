#!/bin/bash
echo "[Info] Mode: $1"

if [ "$1" = "segmentation" ]
then 
    python /home/OCTA-seg/test.py --config_file /home/OCTA-seg/docker/trained_models/ves_seg-S-GAN/config.yml --epoch 30
elif [ "$1" = "generation" ]
then
    python /home/OCTA-seg/generate_vessel_graph.py --config_file /home/OCTA-seg/docker/vessel_graph_gen_docker_config.yml --num_samples $2
elif [ "$1" = "translation" ]
then
    python /home/OCTA-seg/test.py --config_file /home/OCTA-seg/docker/trained_models/GAN/config.yml --epoch 150
else
    echo "Mode $1 does not exist. Choose segmentation, generation or translation."
    exit 1
fi