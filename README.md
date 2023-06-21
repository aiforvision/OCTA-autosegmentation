# Detailed retinal vessel segmentation without human annotations using simulated optical coherence tomography angiographs
This is the repository for the paper [Detailed retinal vessel segmentation without human annotations using simulated optical coherence tomography angiographs (2023)](https://arxiv.org/abs/2306.10941)</b>.

<div style="text-align:center">
    <img src="images/abstract_v4_1.svg">
</div>


## Abstract
Optical coherence tomography angiography (OCTA) is a non-invasive imaging modality that can acquire high-resolution volumes of the retinal vasculature and aid the diagnosis of ocular, neurological and cardiac diseases. Segmentation of the visible blood vessels is a common first step when extracting quantitative biomarkers from these images. Classical segmentation algorithms based on thresholding are strongly affected by image artifacts and limited signal-to-noise ratio. The use of modern, deep learning-based segmentation methods has been inhibited by a lack of large datasets with detailed annotations of the blood vessels. To address this issue, recent work has employed transfer learning, where a segmentation network is trained on synthetic OCTA images and is then applied to real data.
However, the previously proposed simulation models are incapable of faithfully modeling the retinal vasculature and do not provide effective domain adaptation. Because of this, current methods are not able to fully segment the retinal vasculature, in particular the smallest capillaries.
In this work, we present a lightweight simulation of the retinal vascular network based on space colonization for faster and more realistic OCTA synthesis. Moreover, we introduce three contrast adaptation pipelines to decrease the domain gap between real and artificial images. We demonstrate the superior performance of our approach in extensive quantitative and qualitative experiments on three public datasets that compare our method to traditional computer vision algorithms and supervised training using human annotations.
Finally, we make our entire pipeline publicly available, including the source code, pretrained models, and a large dataset of synthetic OCTA images.

# 🔴 TL;DR: Segment my images / Generate synthetic images
We provide a docker file with a pretrained model to segment 3×3 mm² macular OCTA images:
```sh
# Build Docker image. (Only required once)
docker build . -t octa-seg
``` 
To **segment** a set of images replace the placeholders with your directory paths and run:
```sh
docker run -v [DATASET_DIR]:/var/dataset -v [RESULT_DIR]:/var/segmented octa-seg segmentation
``` 
**We provide 500 synthetic training samples** with labels under [./datasets](./datasets). To **generate** _N_ more samples, run:
```sh
docker run -v [RESULT_DIR]:/var/generation octa-seg generation [N]
``` 

# 🔵 Manual Installation
The following section explains how to prepare your environment to run the experiments from the paper, or new experiments. 

### Installation
Make sure you have a clean [conda](https://docs.conda.io/en/main/miniconda.html) environment with python 3 and [pytorch](https://pytorch.org/get-started/locally/) (tested with python 3.10, pytorch==2.0.1, and torchvision==0.15.2). Install the remaining required packages:
 ```sh
pip install -r requirements.txt
 ```


### Synthetic Dataset
We provide 500 synthetic training samples with labels under [./datasets](./datasets). To **create more samples**, visit the respective [README](./datasets/README.md).

### Getting the evaluation datasets 

We use three test datasets:
 - [OCTA-500](https://ieee-dataport.org/open-access/octa-500) (Mingchao Li, Yerui Chen, Songtao Yuan, Qiang Chen, December 23, 2019, "OCTA-500", IEEE Dataport, doi: https://dx.doi.org/10.1109/TMI.2020.2992244. )
 - [ROSE-1](https://imed.nimte.ac.cn/dataofrose.html) (Ma, Yuhui; Hao, Huaying; Xie, Jianyang; Fu, Huazhu; Zhang, Jiong; Yang, Jianlong et al. (2021): ROSE: A Retinal OCT-Angiography Vessel Segmentation Dataset and New Model. In IEEE transactions on medical imaging 40 (3), pp. 928–939. https://doi.org/10.1109/TMI.2020.3042802. )
 - [Giarratano <i>et al.</i>](https://datashare.ed.ac.uk/handle/10283/3528) (Giarratano, Ylenia. (2019). Optical Coherence Tomography Angiography retinal scans and segmentations. University of Edinburgh. Medical School. https://doi.org/10.7488/ds/2729. )


> ⚠️ **_NOTE:_**
> - For the OCTA-500 dataset, make sure to select the correct images and not to include the FAZ segmentation.
> - Each dataset comes with a different level of detail for vessel segmentation. When training on synthetic data, make sure to select the correct min_radius in the repective [config.yml](configs/config_ves_seg-S.yml#L37) for label alignment.
> - When training on synthetic data for the dataset by Giarratano <i>et al.</i>, you have to apply random cropping in the training data augmentations of the [config.yml](configs/config_ves_seg-S.yml#L79) file.

### Getting the pretrained models
We provide a pre-trained GAN model and segmentation model trained for the OCTA-500 dataset under  `./docker/trained_models`.


# How to use
Experiments are organized via config.yml files. We provide several predefined config files under `./configs` for the experiments shown in the paper. Please refer to the respective [README](configs/README.md) for more information.

## GAN training
To re-train a GAN model for the S-GAN experiment in the paper, you can use the provided [config file](./configs/config_gan_ves_seg.yml). The trained Generator is then used for data augmentation when training a separate segmentation network on synthetic data (see [config file](./configs/config_ves_seg-S_GAN.yml)).

```sh
# Train a new Generator network
python train.py --config_file ./configs/config_gan_ves_seg.yml 
```
Now manually copy the path of the generator checkpoint to [./configs/config_ves_seg-S_GAN.yml](./configs/config_ves_seg-S_GAN.yml).
```yml
Train:
    data_augmentation:
        #...
        - name: ImageToImageTranslationd
            model_path: ./results/gan-ves-seg/[FOLDER_NAME]/checkpoints/
```



## Segmentation training
To train models for experiments as shown in the paper, you can use the provided config files under `./configs`. Select the required dataset by specifying the input path in the respective config file. After the training has started, a new folder will be created. The folder contains training details, checkpoints, and a 'config.yml' file that you will need for validation and testing.
```sh
# Start a new training instance
python train.py --config_file ./configs/[CONFIG_FILE_NAME]
```

## Validation
To evaluate trained models (or methods that do not need to be trained), make sure the validation section of the respective config file is correct and run:
```sh
python validate.py --config_file [PATH_TO_CONFIG_FILE] --epoch [EPOCH]
```

## Testing / Inference
To generate segmentations (or transformed images if testing GAN), make sure the test section of the respective config file is correct and run:
```sh
python test.py --config_file [PATH_TO_CONFIG_FILE] --epoch [EPOCH]
```

# 🟢 Citation
If you use this code for your research, please cite our [paper (preprint)](https://arxiv.org/abs/2306.10941):
```bib
@misc{Kreitner2023,
title={Detailed retinal vessel segmentation without human annotations using simulated optical coherence tomography angiographs}, 
author={Linus Kreitner and Johannes C. Paetzold and Nikolaus Rauch and Chen Chen and Ahmed M. Hagag and Alaa E. Fayed and Sobha Sivaprasad and Sebastian Rausch and Julian Weichsel and Bjoern H. Menze and Matthias Harders and Benjamin Knier and Daniel Rueckert and Martin J. Menten},
year={2023},
eprint={2306.10941},
archivePrefix={arXiv},
primaryClass={eess.IV},
url = {https://doi.org/10.48550/arXiv.2306.10941}
}
```

And our [privious work](https://link.springer.com/chapter/10.1007/978-3-031-16452-1_32):
```bib
@InProceedings{Menten2022,
author={Menten, Martin J. and Paetzold, Johannes C. and Dima, Alina
and Menze, Bjoern H. and Knier, Benjamin and Rueckert, Daniel},
title={Physiology-Based Simulation of the Retinal Vasculature Enables Annotation-Free Segmentation of OCT Angiographs},
booktitle={Medical Image Computing and Computer Assisted Intervention -- MICCAI 2022},
year={2022},
publisher={Springer Nature Switzerland},
address={Cham},
pages={330--340},
abstract={Optical coherence tomography angiography (OCTA) can non-invasively image the eye's circulatory system. In order to reliably characterize the retinal vasculature, there is a need to automatically extract quantitative metrics from these images. The calculation of such biomarkers requires a precise semantic segmentation of the blood vessels. However, deep-learning-based methods for segmentation mostly rely on supervised training with voxel-level annotations, which are costly to obtain.},
isbn={978-3-031-16452-1}
}
```