# Deep learning project on classifiaction of images from MNIST dataset

## Description:

This project was made in order to practice application of Tensorflow for building and training neural networks.
I've built CNN and LSTM models using Keras and fitted them with data from the MNIST dataset.

Project takes the next steps:
1. Data standartization.
2. Building and compiling the CNN model.
3. Building and compiling LSTM model.
4. Evaluation of both models and tracking the accuracy metric.

## Dataset:

MNIST dataset from `tensorflow.keras.datasets`

## Development tools:
* Python 3.7
* Tensorflow
* Matplotlib
* Numpy

## Results:

I've got 99.2% accuracy on test set using CNN and 97% accuracy using LSTM.

Overall, CNN's are of course preferred regarding image classification, but LSTM also performed well on this dataset.
It was my first step in deep learning and I'm supposed to go further.

## Installation:
### 1. Clone the repo
    git clone https://github.com/ddzina/drums-classification.git

### 2. Create Anaconda virtual environment:

    conda create --name env --file requirements.txt

Put your name of environment after flag "--name". Created environment will include all the requirement dependencies.
