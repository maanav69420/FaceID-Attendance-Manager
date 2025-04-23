import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

def load_data(data_dir, img_height, img_width):
    images = []
    labels = []
    class_names = os.listdir(data_dir)

    for label in class_names:
        person_dir = os.path.join(data_dir, label)
        if os.path.isdir(person_dir):
            for img_file in os.listdir(person_dir):
                img_path = os.path.join(person_dir, img_file)
                img = load_img(img_path, target_size=(img_height, img_width))
                img_array = img_to_array(img)
                images.append(img_array)
                labels.append(class_names.index(label))  # Use index as label

    return np.array(images), np.array(labels), class_names

def create_model(num_classes):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 3)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model

def train_model():
    img_height, img_width = 200, 200
    data_dir = "dataset"

    # Load dataset
    images, labels, class_names = load_data(data_dir, img_height, img_width)

    # Normalize images
    images = images / 255.0

    # Convert labels to categorical (one-hot encoding)
    labels = to_categorical(labels, num_classes=len(class_names))

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

    # Create the model
    model = create_model(num_classes=len(class_names))

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model
    history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

    # Save the model
    model.save('face_recognition_model.h5')
    print("Model saved as 'face_recognition_model.h5'")

    # Save class names to a file for later use
    with open('class_names.txt', 'w') as f:
        for name in class_names:
            f.write(f"{name}\n")
    print("Class names saved to 'class_names.txt'")


if __name__ == "__main__":
    train_model()
