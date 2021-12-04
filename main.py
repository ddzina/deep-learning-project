
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist         # библиотека базы выборок Mnist
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# стандартизация входных данных
x_train = x_train / 255
x_test = x_test / 255

y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)

x_train = np.expand_dims(x_train, axis=3)
x_test = np.expand_dims(x_test, axis=3)

model = keras.Sequential([
    Conv2D(6, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2), strides=(2, 2)),
    Conv2D(16, (5, 5), strides=(1, 1), padding='same', activation='relu'),
    MaxPooling2D((2, 2), strides=(2, 2)),
    Flatten(),
    Dense(120, activation='relu'),
    Dense(84, activation='linear'),
    #Dropout(0.5),
    Dense(10, activation='softmax')
])

print(model.summary())      # вывод структуры НС в консоль

model.compile(optimizer='adam',
             loss='categorical_crossentropy',
             metrics=['accuracy'])


history = model.fit(x_train, y_train_cat, batch_size=256, epochs=5, verbose=1, validation_split=0.2)

score = model.evaluate(x_test, y_test_cat, verbose=0)

print('Потери при тестировании: ', score[0])
print('Точность при тестировании:', score[1])

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.show()