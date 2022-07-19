import matplotlib.pyplot as plt
import numpy as np

# Load MNIST Dataset
def loadMNIST():
    from keras.datasets import mnist
    (xTrainMNIST, yTrainMNIST), (xTestMNIST, yTestMNIST) = mnist.load_data()

    # Plot MNIST Examples
    n_img = 10
    plt.figure(figsize = (n_img * 2, 2))
    print("10 Random MNIST Sample Images")
    plt.title("10 Random MNIST Sample Images")
    plt.gray()
    for i in range(n_img):
        plt.subplot(1, n_img, i + 1)
        plt.imshow(xTrainMNIST[np.random.randint(0, xTrainMNIST.shape[0])])
    plt.show()

    # Reshape data 
    xTrainMNIST = xTrainMNIST.reshape(xTrainMNIST.shape[0], -1)
    xTestMNIST = xTestMNIST.reshape(xTestMNIST.shape[0], -1)

    print('Training data shape:', xTrainMNIST.shape)
    print('Test data shape:', xTestMNIST.shape)

    return xTrainMNIST, yTrainMNIST, xTestMNIST, yTestMNIST

# EXTRACT CLASSIFIED DATASET
def extractAllClassificationDataset(x, y, numSamples):
    # Numpy arrays to store training set
    x_ = np.zeros((0, 784)) # Image Data
    y_ = np.zeros((0)) # Class Labels

    # numSamples samples per label put in numpy arrays
    for label in range(10):
        tempX = x[y == label]
        tempX = tempX[:numSamples]
        tempY = np.full(numSamples, label)
        
        x_ = np.concatenate((x_, tempX), axis = 0)
        y_ = np.concatenate((y_, tempY), axis = 0)

    return x_.T, y_

# INITIALIZE NEURAL NETWORK
def TwoLayerNetwork(layer_dims=[784, 256, 10]):
    # Initialize Normally Distributed Weights
    W1 = np.random.normal(0, 0.01, (layer_dims[1], layer_dims[0])) # (256, 784)
    W2 = np.random.normal(0, 0.01, (layer_dims[2], layer_dims[1])) # (10, 256)

    # Initialize Biases to 0
    b1 = np.zeros(layer_dims[1]) # (256, )
    b2 = np.zeros(layer_dims[2]) # (10, ))

    # Return array of weights/biases
    params = np.array([W1, b1, W2, b2], dtype=object)
    return params