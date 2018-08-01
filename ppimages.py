import os
from PIL import Image

DATA = './data/'
TRAIN = './train/'
VAL = './val/' 
SIZE = 100,100

def square_up(image):
    image = Image.open(image)
    width,height = image.size[0], image.size[1]
    aspect = width / float(height)
    
    ideal_width, ideal_height = 100, 100
    ideal_aspect = ideal_width / float(ideal_height)
     
    if aspect > ideal_aspect:
        # Then crop the left and right edges:
        new_width = int(ideal_aspect * height)
        offset = (width - new_width) / 2
        resize = (offset, 0, width - offset, height)
    else:
        # ... crop the top and bottom:
        new_height = int(width / ideal_aspect)
        offset = (height - new_height) / 2
        resize = (0, offset, width, height - offset)
     
    square = image.crop(resize).resize((ideal_width, ideal_height), Image.ANTIALIAS)
    return square

def preprocess_images(src, train, val, ratio = .7):
    '''
    splits data into training and validation datasets
    resizes to 100x100 aspect ratio
    '''
    for flower in os.listdir(src):
        images_f = src + flower + '/'
        partition = int(len([file for file in os.listdir(images_f)])*ratio)
        file_num = 0
        for file in os.listdir(images_f):
            image = square_up(images_f+file)
            if file_num < partition:
                location=train
            else:
                location=val
            location += flower+'/'+flower+str(file_num)+'.jpg'
            image.save(location)
            file_num += 1
        file_num = 0

def create_directories():
    os.makedirs('./train/')
    os.makedirs('./val/')
    for flower in ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']:
        os.makedirs('./train/'+flower)
        os.makedirs('./val/'+flower)

if __name__ == "__main__":
    create_directories()
    preprocess_images(DATA, TRAIN, VAL)

