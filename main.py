import tensorflow as tf
import numpy as np


# mnist = tf.keras.datasets.mnist
(x_train, y_train) = (np.load('datasets/x_train.npy'), np.load('datasets/y_train.npy'))
(x_test, y_test) = (np.load('datasets/x_test.npy'), np.load('datasets/y_test.npy'))
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)
model.save('models/model.keras')
