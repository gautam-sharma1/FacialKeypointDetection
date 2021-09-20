# Facial Keypoint Detection 

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

The Facial Keypoint Detection project uses ResNet 51 convolutional neural network (CNN) to find 68 facial keypoints on a face. The keypoints are positioned on the features of a face, such as the eyebrow, nose, and mouth. The points could therefore be used in a number of applications including emotion detection and facial feature classification.

The pretrained ResNet 51 network can be downloaded from my [Google Drive]. 

The following plot shows the training `(blue)` and validation `(orange)` losses respectively 

![alt text](data/Loss.png)

## Installation

- Clone this repository to your local machine.  
- Download the pretrained ResNet 51 model from my [Google Drive]. 

Please make sure to add the downloaded model to the ```saved_models``` folder

### Install Pytorch

> Note: It can be installed using Conda, Pip, Source etc. This repo will show you how to setup your enbvironment using conda. For updated details refer [Pytorch documentation]

For Mac:
```sh
conda install pytorch torchvision torchaudio -c pytorch
```

For Linux:
```sh
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

For Windows:
```sh
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

## Running

```live_inference.py``` uses a webcam to make predictions as follows:


It can be run as
```sh
python3 live_inference.py -cam 1
```
> Note: If you are using a built in camera, give ```0``` as an argument to ```-cam```.

```model.py``` defines another CNN architecture although this architecture is not used in training. 

```main.py``` is the script that trains the model. Normally, you would not need to run this since a pre-trained model is already provided. But, for the curious folks, you can tinker around with the architecture in ```model.py``` and then run ```main.py``` as follows:

```python
python3 main.py -d <path to data folder> -b <BATCH_SIZE> -e <NUMBER_OF_EPOCHS>
```

```plot.py``` defines a class ```Plot``` that is used for plotting throught this package. For example:

```python
    face_dataset = FacialKeypointsDataset(csv_file='./data/training_frames_keypoints.csv',
                                          dataset_location='./data/training')

    # print some stats about the dataset
    print('Length of dataset: ', len(face_dataset))

    p = Plot(face_dataset)  # Instantiate the class

    # plots three random images
    p.plot(num_to_display=3)  
```

```transforms.py``` define transform classes ```Normalize```, ```Rescale```, ```RandomCrop``` and ```ToTensor```. 

```data.py``` defines the class ```FacialKeypointsDataset``` that is used to convert the dataset to a pytorch Dataset.


## Credits
This project is inspired from the ```Udacity Computer Vision Nanodegree```



## License

MIT

**Free Software, Hell Yeah!**

[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)

   [Google Drive]: <https://drive.google.com/file/d/1hriSjRxCN9AjTQLImFNVLX4ndfb_9BXk/view?usp=sharing>
   [Pytorch documentation]: https://pytorch.org
