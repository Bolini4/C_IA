import os
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf
from keras.models import Model, load_model

model = tf.keras.models.load_model('modelme.txt')


layer_outputs = [layer.output for layer in model.layers]
activation_model = Model(inputs=model.input, outputs=layer_outputs)

img = image.load_img("0_1.bmp", target_size=(28, 28), color_mode='grayscale')
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0  # Normalisation

img_array_flattened = img_array.reshape(1, 28 * 28)  # Aplatir l'image
print(img_array_flattened[0])

np.savetxt("imgflatened.txt", img_array_flattened[0])

activations = activation_model.predict(img_array_flattened)




print("-------------------------------------")

for layer_activation in activations:
    print(layer_activation.shape)


# Supposons que "activations" contient les activations du premier étage
first_layer_activations = activations[0]
second_layer_activations = activations[1]
third_layer_activations = activations[2]

# Sauvegarder les activations dans un fichier texte
# np.savetxt('activations_third_layer.txt', first_layer_activations)

sum1 = np.sum(first_layer_activations)
sum2 = np.sum(second_layer_activations)
sum3 = np.sum(third_layer_activations)

print(sum1)
print(sum2)
print(sum3)