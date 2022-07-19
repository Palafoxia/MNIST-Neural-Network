import numpy as np
import matplotlib.pyplot as plt

from functions import *
from initialization import *
from training import *
from evaluation import *

# MAIN ----------------------------------------------------------------------------------
xTrainMNIST, yTrainMNIST, xTestMNIST, yTestMNIST = loadMNIST()

# SELECT DATA
numSamples = 1000
xTrain, yTrain = extractAllClassificationDataset(xTrainMNIST, yTrainMNIST, numSamples)
xTest, yTest = extractAllClassificationDataset(xTestMNIST, yTestMNIST, numSamples)

# TRAIN THE MODEL
# Specify layer dimensions, number of epochs, and learning rate
print("TRAINING MODEL....")
layerDims = [xTrain.shape[0], 256, 10]
epochs = 250
lr = 0.025

# Initialize Neural Network
params = TwoLayerNetwork(layerDims)

# True labels (N x 1)
Y_true = yTrain

# Calculate loss for each epoch
lossHistory = np.zeros(epochs)
for i in range(epochs):
  # Forward Pass
  probs, intermediate = forward(xTrain, params) # Y2 1 x N output, intermediate[X, Z1, Y1, Z2], params[W1, b1, W2, b2]

  # Backpropagation
  grads = backward(Y_true, probs, intermediate, params)

  # Gradient Descent Update
  params = GD(params, grads, lr)

  # Calculate Loss
  lossHistory[i] = MultiClassCrossEntropyLoss(Y_true, probs)
  if i % 25 == 0: # Print loss every 25 epochs
    print("Loss at", i , "epoch:", truncate(lossHistory[i], 4))

# Plot Loss vs Epoches
plt.figure()
plt.plot(lossHistory)
print("Epochs vs Training Loss")
plt.title("Epochs vs Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Training Loss")
plt.show()

# TRAINING ACCURACY
trainingAccuracy, trainingClassifications, correctTrainingIndexes, wrongTrainingIndexes = computeAccuracy(Y_true, probs)
print("TRAINING ACCURACY:", truncate(trainingAccuracy, 4), "%")

# TEST ACCURACY
# Use optimized parameters on a forward pass of the test data set
probs_test = forward(xTest, params)[0]
testAccuracy, testClassifications, correctTestIndexes, wrongTestIndexes = computeAccuracy(yTest, probs_test)
print("TEST ACCURACY:", truncate(testAccuracy, 4), "%")

# Plot five correctly & wrongly classified images from the training and test sets
plotClassifiedImages(5, correctTrainingIndexes, wrongTrainingIndexes, xTrain, "training")
plotClassifiedImages(5, correctTestIndexes, wrongTestIndexes, xTest, "test")

# View more correctly & wrongly classified images from either dataset
while(True):
    choice = input("Would you like to view more images? (y/n) ")
    if choice.lower() == "y":
        
        # Select training or test dataset
        trainingOrTest = input("Which dataset? (training/test) ")
        while(trainingOrTest.lower() != "training" and trainingOrTest.lower() != "test"):
            trainingOrTest = input("Select a valid dataset (training/test) ")

        # Select number of images to plot
        numImages = int(input("How many images would you like to display? "))
        while(numImages < 0 or numImages > 200):
            numImages = int(input("Select a valid number of images (Max = 200) "))

        # Select specific number to plot
        numberChoice = input("Would you like to plot a specific number? (y/n) ")
        while(numberChoice.lower() != "y" and numberChoice.lower() != "n"):
            numberChoice = input("Select a valid choice (y/n) ")

        # Initialization for arrays to use in plotClassifiedImages()
        dataset = np.zeros((784, 0))
        correctIndexes = np.zeros(0)
        wrongIndexes = np.zeros(0)

        # Build new dataset and indexes for 1 number
        if numberChoice.lower() == "y":
            number = int(input("Which number would you like to plot? (0-9) "))
            while(number < 0 or number > 9):
                number = int(input("Select a valid number (0-9) "))

            # Build from training dataset
            if trainingOrTest.lower() == "training":
                currentIndex = 0
                for i in range(correctTrainingIndexes.size):
                    if(yTrain[correctTrainingIndexes[i]] == number):
                        correctIndexes = np.append(correctIndexes, currentIndex)
                        dataset = np.append(dataset, np.vstack(xTrain[:, correctTrainingIndexes[i]]), 1)
                        currentIndex += 1
                for i in range(wrongTrainingIndexes.size):
                    if(yTrain[wrongTrainingIndexes[i]] == number):
                        wrongIndexes = np.append(wrongIndexes, currentIndex)
                        dataset = np.append(dataset, np.vstack(xTrain[:, wrongTrainingIndexes[i]]), 1)
                        currentIndex += 1
                
            # Build from test dataset
            elif trainingOrTest.lower() == "test":
                currentIndex = 0
                for i in range(correctTestIndexes.size):
                    if(yTest[correctTestIndexes[i]] == number):
                        correctIndexes = np.append(correctIndexes, currentIndex)
                        dataset = np.append(dataset, np.vstack(xTest[:, correctTestIndexes[i]]), 1)
                        currentIndex += 1
                for i in range(wrongTestIndexes.size):
                    if(yTest[wrongTestIndexes[i]] == number):
                        wrongIndexes = np.append(wrongIndexes, currentIndex)
                        dataset = np.append(dataset, np.vstack(xTest[:, wrongTestIndexes[i]]), 1)
                        currentIndex += 1
            
        # Use entire training/test dataset
        else: 
            if trainingOrTest.lower() == "training":
                dataset = xTrain
                correctIndexes, wrongIndexes = correctTrainingIndexes, wrongTrainingIndexes
            elif trainingOrTest.lower() == "test":
                dataset = xTest
                correctIndexes, wrongIndexes = correctTestIndexes, wrongTestIndexes

        # PLOT
        plotClassifiedImages(numImages, correctIndexes.astype(np.int64), wrongIndexes.astype(np.int64), dataset, trainingOrTest.lower())
    else:
        print("Exiting program....")
        break