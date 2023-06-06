# Detailed retinal vessel segmentation without human annotations using simulated optical coherence tomography angiographs
This is the repository for the paper <b>"Detailed retinal vessel segmentation without human annotations using simulated optical coherence tomography angiographs" (2023)</b>.

<div style="text-align:center">
    <img src="images/abstract_v4_1.svg">
</div>


## Abstract
Optical coherence tomography angiography (OCTA) is a non-invasive imaging modality that can acquire high-resolution volumes of the retinal vasculature and aid the diagnosis of ocular, neurological and cardiac diseases. In order to automatically extract quantitative biomarkers from these images, the visible blood vessels have to be segmented. Classical segmentation algorithms based on thresholding are strongly affected by image artifacts and limited signal-to-noise ratio. The use of deep learning-based segmentation methods has been inhibited by a lack of large datasets with detailed annotations of the blood vessels. To address this issue, recent work has employed transfer learning, where a segmentation network is trained on synthetic OCTA images and is then applied to real data.
However, previously proposed simulation models are incapable of modeling the entire complexity of retinal vasculature and do not provide effective contrast adaptation. Because of this, current methods are not able to fully segment the retinal vasculature, in particular the smallest capillaries.
In this work, we present a lightweight simulation based on space colonization for faster and more realistic OCTA synthesis. Moreover, we introduce three data transformation pipelines to decrease the domain gap between real and artificial images. We demonstrate the superior performance of our approach in extensive quantitative and qualitative experiments on three public datasets that compare our method to traditional computer vision algorithms, supervised training using human annotations, and previous work using transfer learning.
Finally, we publicly offer our entire pipeline as a tool, with open source code, pretrained models, and a large dataset of synthetic OCTA images.

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
The following section explains how to prepare your environment to run the experiments from the paper or new experiments. 

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
If you use this code for your research, please cite our paper.
```bib
@inproceedings{OCTASeg2023
title={Detailed retinal vessel segmentation without human annotations using simulated optical coherence tomography angiographs},
author={Kreitner, Linus and Paetzold, Johannes C. and Rauch Nikolaus and Chen, Chen and Hagag, Ahmed M. and Fayed, Alaa E. and Sivaprasad, Sobha and Rausch, Sebastian and Weichsel, Julian and Menze, Bjoern H. and Harders, Matthias and Knier, Benjamin and Rueckert, Daniel and Menten, Martin J. },
year={2023}
}
```
