import numpy as np
import tensorflow as tf

# Charger le modèle Keras
model = tf.keras.models.load_model('modelme.txt')

# Extraire les poids et les biais de chaque couche
weights_biases = []
for layer in model.layers:
    layer_weights_biases = layer.get_weights()
    weights_biases.append(layer_weights_biases)

# Afficher les poids et les biais de chaque couche
for i, layer_weights_biases in enumerate(weights_biases):
    layer_weights, layer_biases = layer_weights_biases
    print(f"Layer {i + 1} - Weights shape: {layer_weights.shape}, Biases shape: {layer_biases.shape}")

# Enregistrer les poids et les biais dans des fichiers séparés
for i, layer_weights_biases in enumerate(weights_biases):
    layer_weights, layer_biases = layer_weights_biases
    np.savetxt(f"layer_{i + 1}_weights.txt", layer_weights, fmt='%f')
    np.savetxt(f"layer_{i + 1}_biases.txt", layer_biases, fmt='%f')
