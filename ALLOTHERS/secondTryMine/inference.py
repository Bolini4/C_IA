import os
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model('modelme.txt')

folder_path = '../datas/gutty/'  # Chemin vers le dossier contenant les images

total_images = 0
correct_predictions = 0

for filename in os.listdir(folder_path):
    if filename.endswith('.bmp'):  # Assurez-vous que seuls les fichiers d'images sont traités
        img_path = os.path.join(folder_path, filename)
        img = image.load_img(img_path, target_size=(28, 28), color_mode='grayscale')
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalisation

        img_array_flattened = img_array.reshape(1, 28 * 28)  # Aplatir l'image

        predictions = model.predict(img_array_flattened)
        predicted_class = np.argmax(predictions)

        # Extraction du label à partir du nom du fichier
        label = int(filename.split('_')[0])
        if predictions[0][predicted_class] > 0:
            total_images += 1

            if predicted_class == label:
                correct_predictions += 1



        print("Image :", filename)
        print("Label réel :", label)
        print("Prédiction :", predicted_class)
        print("Confiance :", predictions[0][predicted_class] * 100, "%")
        print("-------------------------------------")

accuracy = correct_predictions / total_images
print("Précision réelle :", accuracy)
print("Nombre total d'images testées :", total_images)
