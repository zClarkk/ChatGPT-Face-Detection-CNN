import tensorflow as tf
import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_lfw_people
from keras.utils import to_categorical

# Load the LFW dataset
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

# Get the features and target values
x = lfw_people.data
y = lfw_people.target

for i, item in enumerate(y):
    y[i] = 1

# Preprocess the data by scaling it to the range [0, 1]
x = x.astype(np.float32) / 255.0

# Split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

x_train = x_train.reshape(-1, 50, 37, 1)
x_test = x_test.reshape(-1, 50, 37, 1)

# Convert the target labels to binary arrays
num_classes = len(np.unique(y))
y_train = to_categorical(y_train, 2)
y_test = to_categorical(y_test, 2)

# Build the model
model = keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fit the model as described in the previous answer
model.fit(x_train, y_train, batch_size=32, epochs=10, steps_per_epoch=len(x_train) // 32)

# Evaluate the model on your test dataset
test_loss, test_acc = model.evaluate(x_test, y_test, batch_size=32)
print('Test accuracy:', test_acc)

