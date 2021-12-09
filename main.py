import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D, LSTM, Embedding, InputLayer

np.random.seed(1337)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# стандартизация входных данных
x_train = x_train / 255
x_test = x_test / 255

y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)

# x_train = np.expand_dims(x_train, axis=3)
# x_test = np.expand_dims(x_test, axis=3)


# инициализация структуры НС

# model = keras.Sequential([
#     Conv2D(6, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu', input_shape=(28, 28, 1)),
#     MaxPooling2D((2, 2), strides=(2, 2)),
#     Conv2D(16, (5, 5), strides=(1, 1), padding='same', activation='relu'),
#     MaxPooling2D((2, 2), strides=(2, 2)),
#     Flatten(),
#     Dense(120, activation='relu'),
#     Dense(84, activation='linear'),
#     # Dropout(0.5),
#     Dense(10, activation='softmax')
# ])

model = keras.Sequential((
    InputLayer(input_shape=(28, 28)),
    LSTM(128),
    Dense(10, activation='softmax')
))

# вывод структуры НС в консоль

print(model.summary())

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# тестирование и вывод точности работы

history = model.fit(x_train, y_train_cat, batch_size=256, epochs=50, verbose=1, validation_split=0.2)

score = model.evaluate(x_test, y_test_cat, verbose=1)

print('Потери при тестировании: ', score[0])
print('Точность при тестировании:', score[1])

# построение графика потерь для анализа переобучения

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.show()
