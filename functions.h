#include <stddef.h>
#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#define ROWS 28
#define COLS 28
#define FLATTENED_SIZE (ROWS * COLS)

#define OUTPUT_SIZE 1092  // Définir la taille de sortie
#define INPUT_SIZE 728  // Remplacer VOTRE_TAILLE_D_ENTREE par la taille de votre entrée


typedef struct {
    float **weights;
    float *biases;
} DenseLayer;

void loadWeightsAndBiases(DenseLayer *layer, const char *weightsFile, const char *biasesFile, int inputSize, int outputSize);

void loadWeightsAndBiasesLayer1(DenseLayer *layer, const char *weightsFile, const char *biasesFile);
void loadWeightsAndBiasesLayer2(DenseLayer *layer, const char *weightsFile, const char *biasesFile);

void flattenImage(unsigned char **image, float flattenedImage[FLATTENED_SIZE]);

float *CalculerFirstLayer64(DenseLayer *layer, float input[]);
float *CalculerSecondLayer1092(DenseLayer *layer, float input[]);
float *CalculerThirdLayer10(DenseLayer *layer, float input[]);

void softmax(float *input, int input_len);
float relu(float x);

float sumVector(float *vector, int size);
float sumVector2D(float **vector, int size, int size2);
float sumVector2DFixed(float **vector, int size, int element);

#endif /* FLATTEN_IMAGE_H */
