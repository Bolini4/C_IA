import numpy as np
import os
from PIL import Image

import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout

#First step is to import images and convert then into numpy arrays.
#Split in like 80% for training and 20% for testing


def load_data(data_dir, num_samples_per_class=10, num_test_samples=3):
    images_train = []
    labels_train = []
    images_test = []
    labels_test = []
    for digit in range(10):
        for sample_idx in range(num_samples_per_class):
            filename = f"{digit}_{sample_idx}.bmp"
            img = Image.open(os.path.join(data_dir, filename))
            img = img.convert('L')
            img = img.resize((28,28)) 
            if sample_idx < num_samples_per_class - num_test_samples:
                images_train.append(img)
                labels_train.append(digit)
            else:
                images_test.append(img) 
                labels_test.append(digit)

    return np.array(images_train), np.array(images_test), np.array(labels_train), np.array(labels_test)




images_train,images_test,labels_train,labels_test = load_data('images')
print(images_train.shape)
print(images_test.shape)

images_validation = images_test[:15]
labels_validation = labels_test[:15]

images_test = images_test[15:]
labels_test = labels_test[15:]






images_test = images_test / 255
images_train = images_train / 255
images_validation = images_validation / 255

# print(images_test[0])


images_train = images_train.reshape((images_train.shape[0], 28*28)).astype('float32')
images_test = images_test.reshape((images_test.shape[0], 28*28)).astype('float32')
images_validation = images_validation.reshape((images_validation.shape[0], 28*28)).astype('float32')

print(images_train.shape)
print(images_test.shape)
print(images_validation.shape)


model = Sequential()

model.add(Dense(64,input_dim = 28*28, activation='relu'))
model.add(Dense(1092,activation='relu'))
model.add(Dense(10, activation='softmax'))

# model.summary()
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(epochs=5,batch_size=1, x=images_train, y=labels_train, validation_data=(images_validation, labels_validation))

predictions = model.predict(images_test)

for i in range(len(predictions)):
    print("Image", i + 1, " - Prédiction:", np.argmax(predictions[i]), " - Vérité:", labels_test[i])
#Sauvegarde du modèle
model.save('modelme.keras')


# Affichage de l'historique de la précision
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()



