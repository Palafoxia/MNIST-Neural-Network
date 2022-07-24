# MNIST NEURAL NETWORK | PYTHON MACHINE LEARNING

This project implements a 2-layer neural network in Python.
The neural network uses the functionality of numpy and matplotlib. 
All additional libraries are included in the virtual environment folder `venv/`.

The goal of the neural network is to classify images in the MNIST digits dataset from the Keras package.
The dataset consists of 10 handwritten digits (0, 1,...., 9) which are of size 28x28 pixels.

The main program initializes and trains a neural network through the use of loss functions, gradients, and optimizers.
Performance and accuracy are evaluated after 250 training epochs on both the training and test datasets.
Examples of correctly and incorrectly classified images from both datasets are displayed on screen.
Then, the program prompts the user if they'd like to view more images.
The user is given  options to choose the dataset (test/training), number of images, and whether they'd like to only view images of a specified digit.

## Running the Application
The application runs in the form of a console application and a Jupyter notebook `MNIST Neural Network.ipynb`.
Make sure to install all necessary libraries or use the included virtual environment `venv/` before running the application.
- tensorflow
- numpy
- matplotlib
- keras

To run the console application, simply run `main.exe`.
You can also compile the source code files manually in the IDE of your choice.
This option displays information in the Python console, but any images displayed must be manually closed before the application can continue.
Also, only 1 "row" of images can be displayed at once (maximum of 10 images).

The Jupyter notebook `MNIST Neural Network.ipynb` contains the same code and details the mathematical concepts underlying the neural network.
You may either follow along and run one code block at a time, or run the entire notebook and view the results.
The Jupyter notebook allows you to interact with the code.
You may experiment with changing the learning rate or number of training epochs to improve performance/efficiency.

## Project Structure
### Source Code
- `evaluation.py` in `src/` contains functions for plotting classified images and computing accuracy
- `functions.py` in `src/` contains the loss and activation functions as well as a helper function to truncate floats
- `initialization.py` in `src/` contains code to load/extract the MNIST dataset and initialize the neural network
- `training.py` in `src/` contains code for forward/back propagation and the gradient descent optimizer
- `main.py` in `src/` runs the main console application

### Other
- `MNIST Neural Network.ipynb` is a Jupyter notebook containing all the previous source code as well as code/mathematical explanations 
- The `venv/` folder contains the virtual environment with the corresponding libraries needed to run the project
- The `bin/` folder contains the necessary files to run `main.exe`

#### Detailed information about the code and neural network can be found in `MNIST Neural Network.ipynb`
