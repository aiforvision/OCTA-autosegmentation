#!/bin/sh
experiment=S_RA_delta_N_gamma_OCTA-500
python /home/linus/repos/OCTA-seg/train.py --config_file /home/linus/repos/OCTA-seg/configs/config_ves_seg-${experiment}.yml --split 0 &&\
python /home/linus/repos/OCTA-seg/train.py --config_file /home/linus/repos/OCTA-seg/configs/config_ves_seg-${experiment}.yml --split 1 &&\
python /home/linus/repos/OCTA-seg/train.py --config_file /home/linus/repos/OCTA-seg/configs/config_ves_seg-${experiment}.yml --split 2 &&\
python /home/linus/repos/OCTA-seg/train.py --config_file /home/linus/repos/OCTA-seg/configs/config_ves_seg-${experiment}.yml --split 3 &&\
python /home/linus/repos/OCTA-seg/train.py --config_file /home/linus/repos/OCTA-seg/configs/config_ves_seg-${experiment}.yml --split 4 &&\
d=($(ls -d /home/linus/repos/OCTA-seg/results/ves-seg-${experiment}/*)) &&\
python /home/linus/repos/OCTA-seg/validate.py --config_file ${d[0]}/config.yml &&\
python /home/linus/repos/OCTA-seg/validate.py --config_file ${d[1]}/config.yml &&\
python /home/linus/repos/OCTA-seg/validate.py --config_file ${d[2]}/config.yml &&\
python /home/linus/repos/OCTA-seg/validate.py --config_file ${d[3]}/config.yml &&\
python /home/linus/repos/OCTA-seg/validate.py --config_file ${d[4]}/config.yml
