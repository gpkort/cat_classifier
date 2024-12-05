import matplotlib.pyplot as plt
import tensorflow as tf
import random

TRAINING_DATA_PATH = './model_dataset/training_data'
VALIDATION_DATA_PATH = './model_dataset/validation_data'

train_ds = tf.data.Dataset.load(TRAINING_DATA_PATH)

# get_label_name = metadata.features['label'].int2str

if __name__ == '__main__':
    # Generate a random integer between 1 and 10 (inclusive)
    for i in range(30):
        number = random.randint(1, 10)
        print(number)
    # image, label = next(iter(train_ds))
    # print(image.shape)
    # _ = plt.imshow(image[0].numpy().astype("uint8"))
    # _ = plt.title("Picture")
    # # print(f"Label: {label.numpy()}")
    # plt.show()