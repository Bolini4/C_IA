import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np



model = tf.keras.models.load_model('mnist.keras')


img_path = 'images/1.png'  # Mettez le chemin de votre image ici
img = image.load_img(img_path, target_size=(28, 28), color_mode='grayscale')
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0  # Normalisation

img_array_flattened = img_array.reshape(1, 28, 28)  # Aplatir l'image


predictions = model.predict(img_array_flattened)
print(predictions)
predicted_class = np.argmax(predictions)

print("Prédiction :", predicted_class)
print("Confiance :", predictions[0][predicted_class] * 100, "%")