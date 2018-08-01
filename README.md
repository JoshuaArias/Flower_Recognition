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
    
after running ppimages.py:
/data
    /daisy
    /dandelion
    /rose
    /sunflower
    /tulip
/train (containing 70% of images for each class)
    /daisy
    /dandelion
    /rose
    /sunflower
    /tulip
/val (remaining 30%)
    /daisy
    /dandelion
    /rose
    /sunflower
    /tulip
    
trained model using cnn_flowers.train()

downloaded test set images using FatKun Batch Download Image Chrome Extension

loaded trained model using cnn_flowers.test_trained_model()

