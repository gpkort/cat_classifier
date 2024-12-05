import tensorflow as tf
from os import path

TEST_DATA_PATH = '../data/model_dataset/test_data'
MODEL_PATH = './keras'
MODEL_NAME = 'pmodel.keras'
MODEL_WEIGHTS_NAME = 'pmodel.weights.h5'

test_dataset = tf.data.Dataset.load(TEST_DATA_PATH)
print(path.join(MODEL_PATH, MODEL_NAME))
model = tf.keras.models.load_model(path.join(MODEL_PATH, MODEL_NAME))

# Evaluate the model
results = model.evaluate(test_dataset)
print(f'Test loss: {results[0]}, Test accuracy: {results[1]}')

# Save the model weights
model.save_weights(path.join(MODEL_PATH, MODEL_WEIGHTS_NAME), overwrite=True)
