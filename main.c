#include <stdio.h>
#include <stdlib.h>
#include <float.h>

#include "Bmp2Matrix.h"
#include "functions.h"

int main(int argc, char* argv[]){

    DenseLayer Layer1, Layer2, Layer3;


    printf("Loading weights and biases...\n");
    //loading of layer 1 OK
    loadWeightsAndBiases(&Layer1, "./weightandbiases/layer_1_weights.txt", "./weightandbiases/layer_1_biases.txt", 784, 64);
    loadWeightsAndBiases(&Layer2, "./weightandbiases/layer_2_weights.txt", "./weightandbiases/layer_2_biases.txt", 64, 1092);
    loadWeightsAndBiases(&Layer3, "./weightandbiases/layer_3_weights.txt", "./weightandbiases/layer_3_biases.txt", 1092, 10);
    printf("Weights and biases loaded successfully\n");


printf("Loading image...\n");

   BMP bitmap;
   FILE* pFichier=NULL;

   pFichier=fopen("0_1.bmp", "rb");     //Ouverture du fichier contenant l'image
   if (pFichier==NULL) {
       printf("%s\n", "0_1.bmp");
       printf("Erreur dans la lecture du fichier\n");
   }
   LireBitmap(pFichier, &bitmap);
   fclose(pFichier);               //Fermeture du fichier contenant l'image

   ConvertRGB2Gray(&bitmap);
//Flatened size defined in the .h (28*28)
   float flatImage[FLATTENED_SIZE];
   flattenImage(bitmap.mPixelsGray, flatImage);


// Put image has /255 like in python to normalize the data
for (int i = 0; i < 784; i++) {
    flatImage[i] = flatImage[i] / 255.000000;
}


    float *output1 = CalculerFirstLayer64(&Layer1, flatImage);
    printf("%f\n", output1[63]);
    //OUTPUT OF LAYER 1 IS LIKE IN PYTHON SAME SUM
    float *output2 = CalculerSecondLayer1092(&Layer2, output1);
    //valeur aberrante sur l'addition


    float *output3 = CalculerThirdLayer10(&Layer3, output2);

float sum2 = 0;


sum2 = sumVector(output2, 1092);

    softmax(output3, 10);

printf("Sum of output2: %.*f\n",DBL_DIG, sum2);

    printf("Output of layer 2: \n");
    for (int i = 0; i < 1092; i++) {
        if (output2[i] > 10000)
        {
            printf("%d\n", i);
            printf("%f\n", output2[i]);
            printf("%f\n", Layer2.biases[i]);
        }
    }

    DesallouerBMP(&bitmap);
    free(output1);
    free(output2);
    free(output3);


    
   return 0;
}