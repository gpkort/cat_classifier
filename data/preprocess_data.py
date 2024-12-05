import tensorflow as tf
from tensorflow.keras import layers
import os
import random


# Set the path to the dataset folder and the desired image size
IMAGE_DIR = './image_data'
DATA_DIR = './model_dataset'
IMAGE_SIZE = (160, 160)  
TRAINING_FILE = 'training_data'
VALIDATION_FILE = 'validation_data'
TEST_FILE = 'test_data'

def augment_image(image_path:str):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3)
    
    augment: int = random.randint(1, 4)
    
    if augment == 1:
        image = tf.image.random_flip_left_right(image)
    elif augment == 2:
        image = tf.image.random_flip_up_down(image)
    elif augment == 3:  
        image = tf.image.random_brightness(image, max_delta=0.5)
    elif augment == 4:
        image = tf.image.random_contrast(image, lower=0.0, upper=1.0)
    return image

# Function to load and resize an image
def load_and_resize_image(image_path, size):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3)
    image = tf.image.resize(image, size)
    return image

def augment_images(image_path:str) -> None:
    for root, dirs, files in os.walk(image_path):
        for i, file in enumerate(files):
            if i % 100 == 0:
                print(f"Processed {i} images...{file}, {i}")
            
            should_augment = random.randint(1, 4)    
            if file.endswith(('jpg', 'jpeg', 'png')) and should_augment == 1:
                new_image = augment_image(os.path.join(root, file))
                tf.io.write_file(os.path.join(root, f"aug_{file}"),
                                 tf.image.encode_jpeg(tf.cast(new_image, tf.uint8)))


# Iterate through all images in the dataset folder and resize them
def resize_all_images(dataset_folder, image_size):
    for root, dirs, files in os.walk(dataset_folder):
        for i, file in enumerate(files):
            if i % 200 == 0:
                print(f"Processed {i} images...{file}")
                
            if file.endswith(('jpg', 'jpeg', 'png')):
                image_path = os.path.join(root, file)
                resized_image = load_and_resize_image(image_path, image_size)
                # Save the resized image back to disk (optional)
                tf.io.write_file(image_path, tf.image.encode_jpeg(tf.cast(resized_image, tf.uint8)))

    print("All images have been resized.")
 

    
def create_datasets(data_dir: str, 
                    image_size: tuple,                    
                    batch_size: int=32, 
                    validation_split: float=0.15, 
                    test_split:float=0.15):
    # Load the dataset and split into training and validation sets
    train_val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        label_mode='categorical',
        validation_split=validation_split + test_split,        
        subset="training", 
        shuffle=True,
        seed=123,
        image_size=image_size,
        batch_size=batch_size
    )

    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
    ])

    train_val_ds = train_val_ds.map(lambda x, y: (data_augmentation(x, training=True), y)) 

    val_test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        label_mode='categorical',
        validation_split=validation_split + test_split,
        subset="validation",
        shuffle=True,
        seed=123,
        image_size=image_size,
        batch_size=batch_size
    )

    # Calculate the number of validation and test samples
    val_size = int(len(val_test_ds) * (validation_split / (validation_split + test_split)))
    # test_size = len(val_test_ds) - val_size

    # Split the validation set into validation and test sets
    val_ds = val_test_ds.take(val_size)
    test_ds = val_test_ds.skip(val_size)

    return train_val_ds, val_ds, test_ds

if __name__ == '__main__':
    # tf.data.experimental.save(dataset, "my_dataset")
    train, validate, test = create_datasets(IMAGE_DIR, IMAGE_SIZE)
    tf.data.Dataset.save(train, os.path.join(DATA_DIR, TRAINING_FILE))
    tf.data.Dataset.save(validate, os.path.join(DATA_DIR, VALIDATION_FILE))
    tf.data.Dataset.save(test, os.path.join(DATA_DIR, TEST_FILE))
    
