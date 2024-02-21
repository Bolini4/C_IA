#include "functions.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <stddef.h>


float sumVector(float *vector, int size) {
    float sum = 0;
    for (int i = 0; i < size; i++) {
        sum += vector[i];
    }
    return sum;
}

float sumVector2D(float **vector, int size, int size2) {
    float sum = 0;
    for (int i = 0; i < size; i++) {
        for(int j = 0; j < size2; j++)
        {
            sum += vector[i][j];
        }
    }
    return sum;
}


float sumVector2DFixed(float **vector, int size, int element) {
    float sum = 0;
    for (int i = 0; i < size; i++) {
        sum += vector[element][i];
    }
    return sum;
}

// ROWS & COLS = 28 (defined in the .hs)
void flattenImage(unsigned char **image, float flattenedImage[FLATTENED_SIZE]) {
    int index = 0;
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            flattenedImage[index++] = (float)image[i][j];
        }
    }
}



float relu(float x) {
    return x > 0 ? x : 0;
}



float *CalculerFirstLayer64(DenseLayer *layer, float input[]) {
    // Allouer dynamiquement de la mémoire pour le vecteur de sortie
    int outputSize = 64;
    int numberOfInput = 784;

    // On alloue la taille de sortie du tableau (64) et on vérifie si l'allocation a bien été faite sinon ça retourne NULL
    float *output = malloc(outputSize * sizeof(float));
    if (output == NULL) {
        exit(1);
    }

    // On passe 64 fois (par rapport aux nombres de lignes)
    for (int i = 0; i < outputSize; i++) {
        output[i] = layer->biases[i];
        // On fait 784 opérations pour calculer la matrice de sortie
        for (int j = 0; j < numberOfInput; j++) {
            output[i] += (layer->weights[j][i] * input[j]); //on ajoute la multiplication matricielle
        }
        // À la fin, on applique la fonction relu
        output[i] = relu(output[i]);
    }

    return output;
}


float *CalculerSecondLayer1092(DenseLayer *layer, float input[]) {
    // Allouer dynamiquement de la mémoire pour le vecteur de sortie
    int outputSize = 1092;
    int numberOfInput = 64;

    //On alloue la taille de sortie du tableau (1092) et on vériifie si l'allocation a bien été faite sino ça retourne NULL
    float *output2 = malloc(outputSize * sizeof(float));
    if (output2 == NULL) {
        exit(1);
    }

    for (int i = 0; i < outputSize; i++) {
        output2[i] = layer->biases[i];
        for (int j = 0; j < numberOfInput; j++) {
            output2[i] += (layer->weights[j][i] * input[j]);
        }
        output2[i] = relu(output2[i]);
    }

    return output2;
}

float *CalculerThirdLayer10(DenseLayer *layer, float input[]) {
    // Allouer dynamiquement de la mémoire pour le vecteur de sortie
    int outputSize = 10;
    int numberOfInput = 1092;
    float *output = malloc(outputSize * sizeof(float));
    if (output == NULL) {
        exit(1);
    }

    for (int i = 0; i < outputSize; i++) {
        output[i] = layer->biases[i];
        for (int j = 0; j < numberOfInput; j++) {
            output[i] += layer->weights[i][j] * input[j];
        }
        //output[i] = relu(output[i]); dont forget to add this
    }

    return output;
}


void loadWeightsAndBiases(DenseLayer *layer, const char *weightsFile, const char *biasesFile, int inputSize, int outputSize) {
    // Charger les poids à partir du fichier weightsFile
    layer->weights = (float**)malloc(outputSize * sizeof(float*));
    for (int i = 0; i < outputSize; i++) {
        layer->weights[i] = (float*)malloc(inputSize * sizeof(float));
    }

    layer->biases = (float*)malloc(outputSize * sizeof(float));

    printf("Loading weights from %s\n", weightsFile);
    FILE *weights_fp = fopen(weightsFile, "r");
    if (weights_fp == NULL) {
        fprintf(stderr, "Erreur lors de l'ouverture du fichier %s\n", weightsFile);
        exit(1);
    }
    
    for (int i = 0; i < outputSize; i++) {
        for (int j = 0; j < inputSize; j++) {
            if (fscanf(weights_fp, "%f ", &layer->weights[i][j]) != 1) {
                fprintf(stderr, "Erreur lors de la lecture des poids\n");
                exit(1);
            }
        }
    }
    
    fclose(weights_fp);

    // Charger les biais à partir du fichier biasesFile
    FILE *biases_fp = fopen(biasesFile, "r");
    if (biases_fp == NULL) {
        fprintf(stderr, "Erreur lors de l'ouverture du fichier %s\n", biasesFile);
        exit(1);
    }
    for (int i = 0; i < outputSize; i++) {
        fscanf(biases_fp, "%f", &layer->biases[i]);
    }
    fclose(biases_fp);
}

void softmax(float *input, int input_len) {
    double max_val = input[0];
    double sum = 0.0;
    
    // Trouver la valeur maximale dans le vecteur d'entrée
    for (int i = 1; i < input_len; i++) {
        if (input[i] > max_val) {
            max_val = input[i];
        }
    }
    
    // Calculer la somme des exp des éléments du vecteur d'entrée
    for (int i = 0; i < input_len; i++) {
        sum += exp(input[i] - max_val);
    }
    
    // Appliquer la formule softmax à chaque élément du vecteur d'entrée
    for (int i = 0; i < input_len; i++) {
        input[i] = exp(input[i] - max_val) / sum;
    }
}


void loadWeightsAndBiasesLayer1(DenseLayer *layer, const char *weightsFile, const char *biasesFile) {
    // Charger les poids à partir du fichier weightsFile
    int BiaisSize = 64;

    int WeightsLines = 784;
    int WeightsColumns = 64; 

    layer->weights = (float**)malloc(WeightsLines * sizeof(float*));
    for (int i = 0; i < WeightsLines; i++) {
        layer->weights[i] = (float*)malloc(WeightsColumns * sizeof(float));
    }

    layer->biases = (float*)malloc(BiaisSize * sizeof(float));

    printf("Loading weights from %s\n", weightsFile);
    FILE *weights_fp = fopen(weightsFile, "r");
    if (weights_fp == NULL) {
        fprintf(stderr, "Erreur lors de l'ouverture du fichier %s\n", weightsFile);
        exit(1);
    }
    
    for (int i = 0; i < WeightsLines; i++) {
        for (int j = 0; j < WeightsColumns; j++) {
            if (fscanf(weights_fp, "%f ", &layer->weights[i][j]) != 1) {
                fprintf(stderr, "Erreur lors de la lecture des poids\n");
                exit(1);
            }
        }
    }
    
    fclose(weights_fp);

    // Charger les biais à partir du fichier biasesFile
    FILE *biases_fp = fopen(biasesFile, "r");
    if (biases_fp == NULL) {
        fprintf(stderr, "Erreur lors de l'ouverture du fichier %s\n", biasesFile);
        exit(1);
    }
    for (int i = 0; i < BiaisSize; i++) {
        fscanf(biases_fp, "%f", &layer->biases[i]);
    }
    fclose(biases_fp);
}

void loadWeightsAndBiasesLayer2(DenseLayer *layer, const char *weightsFile, const char *biasesFile) {
    // Charger les poids à partir du fichier weightsFile
    int BiaisSize = 1092;

    int WeightsLines = 64;
    int WeightsColumns = 1092; 

    layer->weights = (float**)malloc(WeightsLines * sizeof(float*));
    for (int i = 0; i < WeightsLines; i++) {
        layer->weights[i] = (float*)malloc(WeightsColumns * sizeof(float));
    }

    layer->biases = (float*)malloc(BiaisSize * sizeof(float));

    printf("Loading weights from %s\n", weightsFile);
    FILE *weights_fp = fopen(weightsFile, "r");
    if (weights_fp == NULL) {
        fprintf(stderr, "Erreur lors de l'ouverture du fichier %s\n", weightsFile);
        exit(1);
    }
    
    for (int i = 0; i < WeightsLines; i++) {
        for (int j = 0; j < WeightsColumns; j++) {
            if (fscanf(weights_fp, "%f ", &layer->weights[i][j]) != 1) {
                fprintf(stderr, "Erreur lors de la lecture des poids\n");
                exit(1);
            }
        }
    }
    
    fclose(weights_fp);

    // Charger les biais à partir du fichier biasesFile
    FILE *biases_fp = fopen(biasesFile, "r");
    if (biases_fp == NULL) {
        fprintf(stderr, "Erreur lors de l'ouverture du fichier %s\n", biasesFile);
        exit(1);
    }
    for (int i = 0; i < BiaisSize; i++) {
        fscanf(biases_fp, "%f", &layer->biases[i]);
    }
    fclose(biases_fp);
}