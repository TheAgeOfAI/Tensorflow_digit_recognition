import tensorflow as tf
import numpy as np
from main import Models


model = tf.keras.models.load_model('models/model.keras')
(x_test, y_test) = (np.load('datasets/x_test.npy'), np.load('datasets/y_test.npy'))

loss, accuracy = model.evaluate(x_test, y_test, verbose=2)
print('Restored model, accuracy: {:5.2f}%'.format(100 * accuracy))
# print(model.predict(x_test).shape)

model = tf.keras.models.load_model('models/3b1b.keras')

loss, accuracy = model.evaluate(x_test, y_test, verbose=2)
print('Restored model, accuracy: {:5.2f}%'.format(100 * accuracy))
# model.summary()
