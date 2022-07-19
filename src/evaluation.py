import matplotlib.pyplot as plt
import numpy as np
import math

# Plot correctly & wrongly classified images from the dataset
def plotClassifiedImages(n_img, correctIndexes, wrongIndexes, dataset, typeString):
    newDataset = dataset.reshape(28, 28, dataset.shape[1])
    totalRows = int(math.ceil(n_img / 10))

    # Correctly Classified
    print(n_img, "correctly classified images from", typeString, "dataset")
    maxImages = bool(False)
    displayedImages = np.zeros(0)
    for row in range(totalRows): # For each row
        imagesOnRow = min(10, n_img - (row * 10)) # 10 max images per row
        plt.figure(figsize = (imagesOnRow * 2, 2))
        plt.gray()

        # Plot imagesOnRow images
        for i in range(imagesOnRow):
            plt.subplot(1, imagesOnRow, i + 1)
            imageIndex = correctIndexes[np.random.randint(0, correctIndexes.size)]

            # Don't plot images that have already been displayed
            while(imageIndex in displayedImages):
                imageIndex = correctIndexes[np.random.randint(0, correctIndexes.size)]
            displayedImages = np.append(displayedImages, imageIndex)

            # Random correctly classified image
            image = newDataset[:, :, imageIndex]
            plt.imshow(image)

            # No more images left
            if (displayedImages.size == correctIndexes.size):
                maxImages = bool(True)
                break

        plt.show()
        if(maxImages):
            print("Maximum images displayed:", wrongIndexes.size)
            break
        row += 1

    # Wrongly Classified
    print(n_img, "wrongly classified images from", typeString, "dataset")
    maxImages = bool(False)
    displayedImages = np.zeros(0)
    for row in range(totalRows): # For each row
        imagesOnRow = min(10, n_img - (row * 10)) # 10 max images per row
        plt.figure(figsize = (imagesOnRow * 2, 2))
        plt.gray()

        # Plot imagesOnRow images
        for i in range(imagesOnRow):
            plt.subplot(1, imagesOnRow, i + 1)
            imageIndex = wrongIndexes[np.random.randint(0, wrongIndexes.size)]

            # Don't plot images that have already been displayed
            while(imageIndex in displayedImages):
                imageIndex = wrongIndexes[np.random.randint(0, wrongIndexes.size)]
            displayedImages = np.append(displayedImages, imageIndex)

            # Random wrongly classified image
            image = newDataset[:, :, imageIndex]
            plt.imshow(image)

            # No more images left
            if (displayedImages.size == wrongIndexes.size):
                maxImages = bool(True)
                break
            
        plt.show()
        if(maxImages):
            print("Maximum images displayed:", wrongIndexes.size)
            break
        row += 1

# Compute the training/test accuracy and return correct/wrong indexes
def computeAccuracy(Y_true, probs):
    N = probs.shape[1]
    correct = 0
    correctIndexes = np.zeros(0)
    wrongIndexes = np.zeros(0)
    classifications = np.zeros(0)

    for i in range(N):
        classifications = np.append(classifications, np.argmax(probs[:, i]))
        if(classifications[i] == Y_true[i]):
            correct += 1
            correctIndexes = np.append(correctIndexes, i)
        else:
            wrongIndexes = np.append(wrongIndexes, i)
    
    accuracy = (correct/N) * 100
    return accuracy, classifications, correctIndexes.astype(np.int64), wrongIndexes.astype(np.int64)