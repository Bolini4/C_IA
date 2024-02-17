import numpy as np
import os
from PIL import Image

import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout

#First step is to import images and convert then into numpy arrays.
#Split in like 80% for training and 20% for testing


import os
from PIL import Image
import numpy as np

def load_data(data_dir, num_samples_per_class=10, num_test_samples=3):
    images_train = []
    labels_train = []
    images_test = []
    labels_test = []
    images_validation = []
    labels_validation = []
    
    for digit in range(10):
        train_dir = os.path.join(data_dir, 'train')
        test_dir = os.path.join(data_dir, 'test')
        validation_dir = os.path.join(data_dir, 'validation')

        for sample_idx in range(num_samples_per_class):
            # Load training images
            train_filename = f"{digit}_{sample_idx}.bmp"
            train_file_path = os.path.join(train_dir, train_filename)
            if os.path.exists(train_file_path):
                img_train = Image.open(train_file_path).convert('L')
                img_train = img_train.resize((28, 28))
                images_train.append(np.array(img_train))
                labels_train.append(digit)

            # Load testing images
            test_filename = f"{digit}_{sample_idx}.bmp"
            test_file_path = os.path.join(test_dir, test_filename)
            if os.path.exists(test_file_path):
                img_test = Image.open(test_file_path).convert('L')
                img_test = img_test.resize((28, 28))
                images_test.append(np.array(img_test))
                labels_test.append(digit)

            # Load validation images
            validation_filename = f"{digit}_{sample_idx}.bmp"
            validation_file_path = os.path.join(validation_dir, validation_filename)
            if os.path.exists(validation_file_path):
                img_validation = Image.open(validation_file_path).convert('L')
                img_validation = img_validation.resize((28, 28))
                images_validation.append(np.array(img_validation))
                labels_validation.append(digit)

    return np.array(images_train), np.array(images_test), np.array(images_validation), np.array(labels_train), np.array(labels_test), np.array(labels_validation)
    


images_train, images_test, images_validation, labels_train, labels_test, labels_validation = load_data('./images/thomas')

print(images_train.shape)
print(images_test.shape)
print(images_validation.shape)

print("Labels de images_test :", labels_test)
print("Labels de images_validation :", labels_validation)



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

print(images_test[0].shape)

model = Sequential()

model.add(Dense(64,input_dim = 28*28, activation='relu'))
model.add(Dense(1092,activation='relu'))
model.add(Dense(10, activation='softmax'))

# model.summary()
# model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# history = model.fit(epochs=10,batch_size=1, x=images_train, y=labels_train, validation_data=(images_validation, labels_validation))

# predictions = model.predict(images_test)

# for i in range(len(predictions)):
#     print("Image", i + 1, " - Prédiction:", np.argmax(predictions[i]), " - Vérité:", labels_test[i])
# #Sauvegarde du modèle
# model.save('modelme.h5')


# # Affichage de l'historique de la précision
# plt.plot(history.history['accuracy'], label='Training Accuracy')
# plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
# plt.title('Training and Validation Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()



