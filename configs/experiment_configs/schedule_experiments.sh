#!/bin/sh
experiment=ves-seg-S_GAN_OCTA-500
echo $experiment
python /home/linus/repos/OCTA-seg/train.py --config_file /home/linus/repos/OCTA-seg/configs/experiment_configs/config_${experiment}.yml --split 0 --save_latest False --num_workers 16 &&\
python /home/linus/repos/OCTA-seg/train.py --config_file /home/linus/repos/OCTA-seg/configs/experiment_configs/config_${experiment}.yml --split 1 --save_latest False --num_workers 16 &&\
python /home/linus/repos/OCTA-seg/train.py --config_file /home/linus/repos/OCTA-seg/configs/experiment_configs/config_${experiment}.yml --split 2 --save_latest False --num_workers 16 &&\
python /home/linus/repos/OCTA-seg/train.py --config_file /home/linus/repos/OCTA-seg/configs/experiment_configs/config_${experiment}.yml --split 3 --save_latest False --num_workers 16 &&\
python /home/linus/repos/OCTA-seg/train.py --config_file /home/linus/repos/OCTA-seg/configs/experiment_configs/config_${experiment}.yml --split 4 --save_latest False --num_workers 16 &&\

d=($(ls -d /home/linus/repos/OCTA-seg/results/${experiment}/*)) &&\
python /home/linus/repos/OCTA-seg/validate.py --config_file ${d[0]}/config.yml &&\
python /home/linus/repos/OCTA-seg/validate.py --config_file ${d[1]}/config.yml &&\
python /home/linus/repos/OCTA-seg/validate.py --config_file ${d[2]}/config.yml &&\
python /home/linus/repos/OCTA-seg/validate.py --config_file ${d[3]}/config.yml &&\
python /home/linus/repos/OCTA-seg/validate.py --config_file ${d[4]}/config.yml
