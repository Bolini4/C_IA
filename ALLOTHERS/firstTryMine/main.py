import numpy as np
import os
from PIL import Image

import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

#First step is to import images and convert then into numpy arrays.
#Split in like 80% for training and 20% for testing


def load_data(data_dir, num_samples_per_class=10, num_test_samples=2):
    images_train = []
    labels_train = []
    images_test = []
    labels_test = []
    for digit in range(10):
        for sample_idx in range(num_samples_per_class):
            filename = f"{digit}_{sample_idx}.bmp"
            img = Image.open(os.path.join(data_dir, filename))
            img = img.convert('L')
            img = img.resize((64, 64)) 
            if sample_idx < num_samples_per_class - num_test_samples:
                images_train.append(img)  # Conserver les trois canaux
                labels_train.append(digit)
            else:
                images_test.append(img)  # Conserver les trois canaux
                labels_test.append(digit)

    return np.array(images_train), np.array(images_test), np.array(labels_train), np.array(labels_test)




images_train,images_test,labels_train,labels_test = load_data('images')
print(images_train.shape)
print(images_test.shape)



images_test = images_test / 255
images_train = images_train / 255

plt.figure(figsize=(10, 10))
for i in range(20):
    plt.subplot(4, 5, i + 1)
    plt.xticks([])  # Supprimer les graduations sur l'axe des x
    plt.yticks([])  # Supprimer les graduations sur l'axe des y
    plt.grid(False)  # Désactiver la grille
    plt.imshow(images_train[i], cmap=plt.cm.binary)  # Afficher l'image en niveaux de gris
    plt.xlabel(labels_train[i])  # Ajouter l'étiquette comme xlabel
plt.show()
# print(images_test[0])


images_train = images_train.reshape((images_train.shape[0], 64*64)).astype('float32')
images_test = images_test.reshape((images_test.shape[0], 64*64)).astype('float32')

print(images_train.shape)
print(images_test.shape)


model = Sequential()

model.add(Dense(64,input_dim = 64  * 64, activation='relu'))
model.add(Dense(6000,activation='relu'))
model.add(Dense(20, activation='softmax'))

# model.summary()
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(epochs=25,batch_size=1, x=images_train, y=labels_train, validation_data=(images_test, labels_test))

scores = model.evaluate(images_test, labels_test)
print(scores)
print('Accuracy: {}'.format(scores[1]))




