import tensorflow as tf
from os import path, walk
from PIL import Image
import numpy as np

IMAGES_DIR = "./data/images/"
TEST_DATA_PATH = './data/model_dataset/test_data'
MODEL_PATH = './model/keras'
MODEL_NAME = 'pmodel.keras'
IMAGE_SIZE = (160, 160)

model = tf.keras.models.load_model(path.join(MODEL_PATH, MODEL_NAME))

def jpg_to_numpy_array(image_path: str) -> np.ndarray:  
    img = tf.io.read_file(image_path)  
    img = tf.image.decode_jpeg(img, channels=3)    
    img_array = img.numpy()  
    
    return img_array

def load_and_resize_image(image_path, size)-> np.ndarray:
    image = tf.keras.utils.load_img(image_path, target_size=size)
    input_arr = tf.keras.utils.img_to_array(image)
    return np.array([input_arr]) 

    
    
if __name__ == '__main__':
    for root, dirs, files in walk(IMAGES_DIR):
        for file in files:
            arr = load_and_resize_image(path.join(root, file), IMAGE_SIZE) 
            # print(f"Prediction for {file}: {arr}")
            val = model.predict(arr)
            print(f'Prediction for {file}: {val}')