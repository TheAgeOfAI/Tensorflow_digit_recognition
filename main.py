import tensorflow as tf
import numpy as np


class Models:
    def __init__(self):
        # load data from internet.
        # mnist = tf.keras.datasets.mnist

        # load data from datasets folder.
        self.x_train, self.y_train = (np.load('datasets/x_train.npy'), np.load('datasets/y_train.npy'))
        self.x_test, self.y_test = (np.load('datasets/x_test.npy'), np.load('datasets/y_test.npy'))
        self.greyscale_dataset()

    def greyscale_dataset(self):
        self.x_train, self.x_test = self.x_train / 255.0, self.x_test / 255.0

    def tensorflow_model(self):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation='softmax')
        ])

        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        model.fit(self.x_train, self.y_train, epochs=5)
        model.evaluate(self.x_test, self.y_test)
        model.save('models/model.keras')

    def three_blue_model(self):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])

        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        model.fit(self.x_train, self.y_train, epochs=5)
        model.evaluate(self.x_test, self.y_test)
        model.save('models/3b1b.keras')


if __name__ == "__main__":
    models = Models()
    print("Choose which model to create:")
    print("1. Tensorflow official model.")
    print("2. 3blue1brown model")
    choice = int(input("Choice :"))
    if choice == 1:
        models.tensorflow_model()
    elif choice == 2:
        models.three_blue_model()
    else:
        exit(0)
