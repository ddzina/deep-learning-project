import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D, LSTM, Embedding, InputLayer

np.random.seed(1337)

# импорт данных из датасета

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# стандартизация входных данных

x_train = x_train / 255
x_test = x_test / 255

y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)

LeNet_x_train = np.expand_dims(x_train, axis=3)
LeNet_x_test = np.expand_dims(x_test, axis=3)

# инициализация структуры НС

LeNet5 = keras.Sequential([
    Conv2D(6, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2), strides=(2, 2)),
    Conv2D(16, (5, 5), strides=(1, 1), padding='same', activation='relu'),
    MaxPooling2D((2, 2), strides=(2, 2)),
    Flatten(),
    Dense(120, activation='relu'),
    Dropout(0.4),
    Dense(84, activation='linear'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

LSTMRNN = keras.Sequential((
    InputLayer(input_shape=(28, 28)),
    LSTM(128, return_sequences=True),
    LSTM(64),
    Dense(10, activation='softmax')
))

# вывод структуры НС в консоль

print(LeNet5.summary())
print(LSTMRNN.summary())

# сборка НС

LeNet5.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

LSTMRNN.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# тестирование и вывод точности работы

LSTMHistory = LSTMRNN.fit(x_train, y_train_cat, batch_size=128, epochs=25, verbose=1, validation_split=0.2)
LeNetHistory = LeNet5.fit(LeNet_x_train, y_train_cat, batch_size=128, epochs=25, verbose=1, validation_split=0.2)

LSTMScore = LSTMRNN.evaluate(x_test, y_test_cat, verbose=1)
LeNetScore = LeNet5.evaluate(LeNet_x_test, y_test_cat, verbose=1)

print('Потери при тестировании LeNet-5: ', LeNetScore[0])
print('Точность при тестировании LeNet-5:', LeNetScore[1])

print('Потери при тестировании LSTM: ', LSTMScore[0])
print('Точность при тестировании LSTM:', LSTMScore[1])

# построение графика потерь для анализа переобучения

plt.plot(LeNetHistory.history['loss'], color='green')
plt.plot(LeNetHistory.history['val_loss'], color='red')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss per epoch')
plt.show()

plt.plot(LSTMHistory.history['loss'], color='green')
plt.plot(LSTMHistory.history['val_loss'], color='red')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss per epoch')
plt.show()

# построения графика зависимсти точности в течение обучения

plt.plot(LeNetHistory.history['accuracy'], color='green')
plt.plot(LSTMHistory.history['accuracy'], color='red')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy per epoch')
plt.show()