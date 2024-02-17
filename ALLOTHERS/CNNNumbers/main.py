import os
import cv2

from keras.datasets import mnist

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

import matplotlib.pyplot as plt
import numpy as np


(x_train, y_train),(x_test,y_test) = mnist.load_data()

x_train = x_train.reshape((x_train.shape[0],28,28,1)).astype('float32') / 255
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1)).astype('float32') / 255
#Au niveau des paramètres : x_trainshapte[0] t-> Nombre d'échantillons
#28,28 -> largeur et hauteur de l'image
#1 -> c'est les canaux d'images (dans notre cas NB y'a un canal, dans le RVB c'est 3)
#et le /255 c'erst pour avoir des valeurs entre 0 et 1 (accélerer la vitesse)

model = Sequential()


#Création du modèle

model.add(Conv2D(32,(3,3), activation='relu', input_shape=(28,28,1)))
model.add(MaxPooling2D((2, 2)))


model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, )))


# Aplatir les données pour la couche dense
model.add(Flatten())

# Couche dense
model.add(Dense(64, activation='relu'))

# Couche de sortie
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()
history = model.fit(epochs=2,batch_size=64, x=x_train, y=y_train, validation_data=(x_test, y_test))

scores = model.evaluate(x_test, y_test)
print(scores)
print('Accuracy: {}'.format(scores[1]))

model.save('mnist.keras')

# # Affichage de l'historique de la précision
# plt.plot(history.history['accuracy'], label='Training Accuracy')
# plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
# plt.title('Training and Validation Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()

# # Affichage de l'historique de la perte
# plt.plot(history.history['loss'], label='Training Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.title('Training and Validation Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()

#Training sur mes images

# folder_path = 'images/'
# image_files = os.listdir('images/')
# print(image_files)

# for image_file in image_files:
#     # Charger l'image
#     image_path = os.path.join(folder_path, image_file)
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Charger l'image en niveaux de gris
#     image = cv2.resize(image, (28, 28))  # Redimensionner l'image à la taille attendue par le modèle
#     image = image.astype('float32') / 255  # Normaliser les valeurs de pixels

#     plt.imshow(image, cmap='gray')
#     plt.title('Image prétraitée')
#     plt.axis('off')  # Désactiver les axes
#     plt.show()
      
#     # Ajouter une dimension pour correspondre à la forme attendue par le modèle
#     image = np.expand_dims(image, axis=0)
#     image = np.expand_dims(image, axis=-1)


#     # Faire une prédiction avec le modèle
#     prediction = model.predict(image)
    
#     # Interpréter les résultats (par exemple, afficher la classe prédite)
#     predicted_class = np.argmax(prediction)
#     print("Image:", image_file, "- Classe prédite:", predicted_class)