import matplotlib.pyplot as plt
import numpy as np
from os import path
import tensorflow as tf

TRAINING_DATA_PATH = '../data/model_dataset/training_data'
VALIDATION_DATA_PATH = '../data/model_dataset/validation_data'
TEST_DATA_PATH = '../data/model_dataset/test_data'
BATCH_SIZE = 32
IMG_SIZE = (160, 160)
AUTOTUNE = tf.data.AUTOTUNE
MODEL_PATH = './keras'



preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

training_data:tf.data.Dataset = tf.data.Dataset.load(TRAINING_DATA_PATH)
validation_data:tf.data.Dataset = tf.data.Dataset.load(VALIDATION_DATA_PATH)
test_dataset = tf.data.Dataset.load(TEST_DATA_PATH)

training_data.prefetch(AUTOTUNE)
validation_data.prefetch(AUTOTUNE)
test_dataset.prefetch(AUTOTUNE)

def create_model() -> tf.keras.Model:
    IMG_SHAPE = IMG_SIZE + (3,)
   
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')
    base_model.trainable = False
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    prediction_layer = tf.keras.layers.Dense(2, activation='sigmoid')
    
    
    inputs = tf.keras.Input(shape=(160, 160, 3))
    x = preprocess_input(inputs)
    x = base_model(x, training=False)
    x = global_average_layer(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = prediction_layer(x)
    
    return tf.keras.Model(inputs, outputs)

if __name__ == "__main__":
   model = create_model()
   initial_epochs = 20
   
   base_learning_rate = 0.0001
   model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.5, name='accuracy')])
   
   history = model.fit(training_data,
                        epochs=initial_epochs,
                        validation_data=validation_data)
   
   model.save(path.join(MODEL_PATH, 'pmodel.keras'))
   print("Model saved.")    
   
   
   

