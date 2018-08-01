# Flower_Recognition
Implemented CNN to classify 5 different types of flowers

data set: https://www.kaggle.com/alxmamaev/flowers-recognition#flowers.zip

original file structure:

/data

    /daisy
    
    /dandelion
    
    /rose
    
    /sunflower
    
    /tulip
    
split data into training and validation sets and fixed aspect ratio using ppimages.preprocess_images()
    
trained model using cnn_flowers.train()

downloaded test set images using FatKun Batch Download Image Chrome Extension

loaded trained model using cnn_flowers.test_trained_model()

