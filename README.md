# pytorch_image_classifier
##Image Classifier for flowers based on pre-trained networks


### Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
3. [Project Motivation](#motivation)
4. [File Descriptions](#files)
5. [Limitations](#limitations)
6. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>
The code in this project is written in Python 3.6.6 :: Anaconda custom (64-bit).  
The following additional libraries have been used:
- sklearn
- IPython.display
- visuals (supplied)


## Usage <a name="usage"></a>
usage: train.py [-h] [--save_dir SAVE_DIR] [--arch  {vgg11,vgg13,vgg16,vgg19}]
                [--learning_rate LEARNING_RATE] [--hidden_units HIDDEN_UNITS]
                [--epochs EPOCHS] [--gpu]
                data_dir
                
usage: predict.py [-h] [--topk TOPK] [--category_names  CATEGORY_NAMES]
                  [--gpu]
                  input checkpoint
                  

## Project Motivation<a name="motivation"></a>
In this project we trained an image classifier to recognize different species of flowers.
The project is broken down into multiple steps:
* Load and preprocess the image dataset
* Train the image classifier on your dataset
* Use the trained classifier to predict image content
We train 102 different types of flowers, where there ~20 images per flower to train on. Then we  use the trained classifier to see if we can predict the type for new images of the flowers.



## File Descriptions <a name="files"></a>
The files included in this project are:
* Image Classifier Project.ipynb, Jupyter notebook with the python code for the whole application
* train.py, command line application that loads a pre-trained network ('vgg13', 'vgg11', 'vgg13', 'vgg16', 'vgg19') and trains a classifier on the images in the "train", "valid" and "test" data directories. Then it saves the checkpoint of the trained network
* predict.py, command line application that predicts class if images in a image directory:
* cat_to_name.json, JSON object that gives a dictionary mapping the integer encoded categories to the actual names of the flowers.


## Limitations<a name="limitations"></a>
The data for this project is quite large - in fact, it is so large that I have not uploaded it onto Github.  If you would like the data for this project, you will have it to download [here](https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz). Also bear in mind that the code will be very slow to run on your local machine unless you have a GPU.  

## Licensing, Authors, Acknowledgements<a name="licensing"></a>
MIT License

Copyright (c) 2018 Udacity

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


