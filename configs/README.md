# Config Files

This folder contains a set of predefined config files, as well as a template to create your own custom files.

## What is a config file?
In this repository we use config files to simplify the configuration of training, validation and testing. Each config file encodes all necessary information about the used datasets, model, model architecture, data pre- and postprocessing, and output folders. When a new model is trained, the respective config file is copied to the output folder and filled with additional information. This ensure that one can always see how a model was trained.

## How do I use a config file?
We provide an individual config file for each experiment in the paper. In order to use it, you need to update the source paths of:
- Training dataset
- Validation dataset
- Test Dataset
- Output path
- Model path (in some cases)

If you are training on synthetic data, you should also update the `min_radius` property, depending on the dataset. Check the [provided template config file](config_template.yml) for more information.