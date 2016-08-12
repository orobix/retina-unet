# Retina blood vessel segmentation with convolution neural network

This repository contains the implementation of a convolutional neural network used to segment blood vessels in retina fundus images. This is a binary classification task: the neural network predicts if each pixel in the fundus image is either a vessel or not.  
The neural network structure is derived from the *U-Net*, described in this [paper](https://arxiv.org/pdf/1505.04597.pdf).  
The performance of this neural network is tested on the DRIVE database, and it achieved the best score in terms of area under the ROC in comparison to the other methods published so far.


## Methods
Before the training, the 20 images of the DRIVE training datasets are pre-processed with the following transformations:
- Gray-scale conversion
- Standardization
- Contrast-limited adaptive histogram equalization (CLAHE)
- Gamma adjustment

The training of the neural network is performed on sub-images (called patches) of the pre-processed full images. Each patch, of dimension 48x48, is obtained by randomly select its center point inside the full image. Also the patches partially or completely outside the FOV are selected, in this way the neural network learns to discriminate the FOV border from the blood vessels.  
A set of 175000 patches are randomly selected for the training, equally split among the 20 images (8750 patches per image). Although the patches overlap, i.e. different patches may contain same part of the original images, no further data augmentation is performed.

The neural network architecture is derived from the *U-net* concept (see the [paper](https://arxiv.org/pdf/1505.04597.pdf)). Compared to the original design, in this case all the patch pixels are labeled and compared to the ground truth, not only the patch center.
The loss function is the cross-entropy and the stochastic gradient descent is employed. The activation function after each convolutional layer is the Rectifier Linear Unit (ReLU), and a dropout of 0.2 is inserted between two consecutive convolutional layers.  
The training is performed for 150 epochs, with a mini-batch size of 32 patches. After each epoch, the model is validated against all patches (no overlap) forming the 20 images of the DRIVE testing dataset, after they have been pre-processed in the same way as for the training images.


## Results
The test is performed with the 20 images of the DRIVE testing dataset, using the gold standard as ground truth.  
The results reported in the `./test` folder are referred to the trained model which reported the minimum validation loss. Only the pixels belonging to the FOV are considered. The FOV is identified with the masks included in the DRIVE database.

In order to improve the performance, the vessel probability of each pixel is obtained by averaging multiple predictions. With a stride of 5 pixels in both height and width, multiple consecutive overlapping patches are extracted in each testing image. Finally, for each pixel, the vessel probability is obtained by averaging over all the predicted patches covering the pixel.

The files `test_Original_GroundTruth_PredictionX.png` show, from top to bottom, the original pre-processed image, the ground truth and the prediction. In the predicted image, each pixel show the vessel predicted probability, no threshold is applied.

The following table compares the area under the ROC curve (AUC ROC) to other methods published so far:

| Tables              | AUC ROC on DRIVE |
| --------------------|:----------------:|
| Jiang et al         | .9327            |
| Soares et al        | .9614            |   
| Osareh et al        | .9650            |    
| Azzopardi et al.    | .9614            |
| Roychowdhury et al. | .9670            |
| Fraz et al.         | .9747            |
| Qiaoliang et al.    | .9738            |
| Melinscak et al.    | .9749            |
| Liskowski et al.^   | .9790            |
| **this method**    | **.9791**        |

^ different definition of FOV


## Run experiment on DRIVE
The code is written in Python, it is possible to replicate the experiment on the DRIVE database by following the guidelines below.


### Prerequisities
The neural network is developed with the Keras library, we refer to the [Keras library](https://github.com/fchollet/keras) for the installation.

The following dependencies are needed:
- numpy >= 1.11.1
- PIL >=1.1.7
- opencv >=2.4.10
- h5py >=2.6.0
- ConfigParser >=3.5.0b2
- scikit-learn >= 0.17.1


Also, you need the DRIVE database, which can be freely downloaded as explained in the next section.

### training

First of all, you need the DRIVE database. We are not allowed to provide the data here, but you can download the DRIVE database at the official [website](http://www.isi.uu.nl/Research/Databases/DRIVE/). Extract the images to a folder, and call it "DRIVE", for example. This folder should have the following tree:
```
DRIVE
│
└───test
|    ├───1st_manual
|    └───2nd_manual
|    └───images
|    └───mask
│
└───training
    ├───1st_manual
    └───images
    └───mask
```
We refer to the DRIVE website for the description of the data.

It is convenient to create hdf5 datasets of the ground truth, masks and images for both the training and testing.
In the root folder, just run:
```
python prepare_datasets_DRIVE.py
```
The hdf5 datasets for training and testing will be created in the folder `./DRIVE_datasets_training_testing/`
N.B: If you gave a different name for the DRIVE folder, you need to specify it in `the prepare_datasets_DRIVE.py` file.

Now we can configure the experiment, All the settings can be specified in the file `configuration.txt`, divided in the following sections:  
**[data paths]**  
Change these paths only if you have modified the `prepare_datasets_DRIVE.py` file  
**[experiment name]**  
Choose a name for the experiment, a folder with the same name will be created and will contain all the results and the trained neural networks.  
**[data attributes]**  
The network is trained on sub-images (patches) of the original full images, specify here the dimension of the patches.   
**[training settings]**  
Here you can specify:  
- *N_subimgs*: total number of patches randomly extracted from the original full images. This number must be a multiple of 20, since an equal number of images is extracted in each of the 20 original training images.
- *inside_FOV*: choose if the patches must be selected inside the Field Of View (FOV), i.e. they don't contains the border mask. The neural network correctly learns to exclude the FOV border if also the patches in the border mask are included. However, a higher number of patches are required.
- *N_epochs*: Number of training epochs.
- *batch_size*: mini batch size.
- *full_images_to_test*: Number of full images for validation, max 20. The testing dataset is used also for validation during training, the full images are divided in patches, but not randomly and with no overlap.
- *nohup*: the standard output during the training is redirected and saved in a log file.


After all the parameters have been configured, you can run the training by:
```
python run_training.py
```
If available, a GPU will be used.  
The following files will be saved in the folder with the same name of the experiment:
- model architecture (json)
- picture of the model structure (png)
- a copy of the configuration file
- model weights at last epoch (hdf5)
- model weights at best epoch (hdf5)
- sample of the training patches and their corresponding ground truth (png)
- sample of the testing patches and their corresponding ground truth (png)


### Evaluate the trained model
The performance of the trained model is evaluated against the DRIVE testing dataset, consisting of 20 images (as many as in the training set).

The parameters for the testing can be tuned again in the `configuration.txt` file, specifically in the [testing settings] section, as described below:  
**[testing settings]**  
- *best_last*: choose the model for the testing prediction. best = the model with the lowest loss obtained during the training; last = the model at the last epoch
- *full_images_to_test*: Number of full images for testing, max 20.
- *N_group_visual*: choose how many images per row in the saved figures.
- *average_mode*: If true, the predicted vessel probability for each pixel is computed by averaging the predicted probability over multiple overlapping patches covering the same pixel.  
- *stride_height*: Relevant only if average_mode is True. The stride along the height for the overlapping patches, smaller stride gives higher number of patches.
- *stride_width*: same as stride_height.
- *nohup*: the standard output during the prediction is redirected and saved in a log file.

The section **[experiment name]** must have the name of the experiment you want to test, while **[data paths]** contains the paths to the testing datasets. The section **[training settings]** will be ignored.

Run the testing by:
```
python run_testing.py
```
If available, a GPU will be used.  
The following files will be saved in the folder with same name of the experiment:
- The ROC curve  (png)
- The Precision-recall curve (png)
- Picture of all the testing pre-processed images (png)
- Picture of all the corresponding segmentation ground truth (png)
- Picture of all the corresponding segmentation predictions  (png)
- One or more pictures including (top to bottom): original pre-rocessed image, ground truth, prediction.
- report of the performance.

All the results are referred only to the pixels belonging to the FOV, selected by the masks included in the DRIVE database


## License

This project is licensed under the MIT License 

Copyright Daniele Cortinovis Orobix s.r.l.
