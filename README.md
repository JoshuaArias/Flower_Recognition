# Flower_Recognition
Implemented CNN to classify 5 different types of flowers, 89% training accuracy, 65% validation accuracy

data set: https://www.kaggle.com/alxmamaev/flowers-recognition#flowers.zip

file structure after download:

    /data

        /daisy
    
        /dandelion
    
        /rose
    
        /sunflower
    
        /tulip
    
1. Split data into training and validation sets and fixed aspect ratio using ppimages.preprocess_images()
    
2. Trained model using cnn_flowers.train()

3. Downloaded test set images using FatKun Batch Download Image Chrome Extension

4. Loaded trained model using cnn_flowers.test_trained_model()

