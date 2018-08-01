from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.preprocessing import image
import keras.optimizers
import numpy as np
import os

img_width, img_height = 100,100
train_data_dir = './train/'
validation_data_dir = './val/'
nb_train_samples = 3023
nb_validation_samples = 1300
epochs = 50
batch_size = 16

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

def create_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(5))
    model.add(Activation('softmax'))
    
    return model

def train():
    model = create_model()
    optimizer = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='categorical_crossentropy', 
                  optimizer=optimizer, 
                  metrics=['accuracy'])
    
    
    train_datagen = ImageDataGenerator()
    
    train_generator = train_datagen.flow_from_directory(
        directory=train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')
    
    validation_generator = train_datagen.flow_from_directory(
        directory=validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')
    
    model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size)
    
    weight_number = len([file for file in os.listdir('./weights/')])+1
    model.save_weights('weight_{}.h5'.format(weight_number))

def read_prediction(result, file, path):
    import operator
    index, value = max(enumerate(result), key=operator.itemgetter(1))
    flowers = {
        '0':'daisy',
        '1':'dandelion',
        '2':'rose',
        '3':'sunflower',
        '4':'tulip'}
    out = "predicted: {:10}, from file: {:10}, with confidence: {:4f}".format(flowers[str(index)], file, value)
    if flowers[str(index)] == file:
        print('.', end ='')
        return 0, value
    else:
        print('!', end = '')
        return 1, value

def run_tests(model):
    for flower in os.listdir('./test/'):
        count, total_failed, conf = 0,0,0
        for file in os.listdir('./test/'+flower):
            count += 1
            test_image = image.load_img('./test/'+flower+'/'+file, target_size =(100,100))
            test_image = image.img_to_array(test_image)
            test_image = np.expand_dims(test_image, axis = 0)
            result = model.predict(test_image)
            t, c =  read_prediction(result[0], flower,'./test/'+flower+'/'+file)
            total_failed += t
            conf += c
        out = "\n{}: failed {} out of {} ({:4f} accuracy)\n".format(flower,total_failed,count,1-total_failed/count)
        print(out)
        
def test_trained_model():
    model = create_model()
    model.load_weights('./weights/weight_1.h5')
    run_tests(model)

if __name__ == "__main__":
    test_trained_model()
    
    