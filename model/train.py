import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt
from os import path

TRAINING_DATA_PATH = '../data/model_dataset/training_data'
VALIDATION_DATA_PATH = '../data/model_dataset/validation_data'
MODEL_PATH = './h5'

training_data = tf.data.Dataset.load(TRAINING_DATA_PATH)
validation_data = tf.data.Dataset.load(VALIDATION_DATA_PATH)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(2, activation='softmax')  # Two categories
])

def train_model(model, training_data, validation_data):       
    
    history = model.fit(training_data, validation_data=validation_data, epochs=75)
    
    # Plot training & validation accuracy values
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()
    
    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

if __name__ == "__main__":
    print("Training model...")
    model.compile(optimizer='adam', 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    print("Model compiled.")
    model.summary()
    print("Starting training...")
    train_model(model, training_data, validation_data)
    print("Training complete.")
    model.save(path.join(MODEL_PATH, 'model.keras'))
    # model.save('my_model.keras')
    print("Model saved.")    
    
    
    