#include <stdio.h>
#include <stdlib.h>
#include <float.h>

#include "Bmp2Matrix.h"
#include "functions.h"

int main(int argc, char* argv[]){

    DenseLayer Layer1, Layer2, Layer3;


    printf("Loading weights and biases...\n");
    printf("JE SUIS LA\n");
    //loading of layer 1 OK
    // loadWeightsAndBiases(&Layer1, "./weightandbiases/layer_1_weights.txt", "./weightandbiases/layer_1_biases.txt", 784, 64);
    // loadWeightsAndBiases(&Layer2, "./weightandbiases/layer_2_weights.txt", "./weightandbiases/layer_2_biases.txt", 64, 1092);
    // loadWeightsAndBiases(&Layer3, "./weightandbiases/layer_3_weights.txt", "./weightandbiases/layer_3_biases.txt", 1092, 10);

    loadWeightsAndBiasesLayer1(&Layer1, "./weightandbiases/layer_1_weights.txt", "./weightandbiases/layer_1_biases.txt");
    loadWeightsAndBiasesLayer2(&Layer2, "./weightandbiases/layer_2_weights.txt", "./weightandbiases/layer_2_biases.txt");
    printf("Weights and biases loaded successfully\n");
    printf("%f\n", Layer2.weights[62][1090]);
    


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
//DANS CETTE CONFIGURATION DE CHARGEMENT, LE LAYER 1 EST OK!!!!
float *output2 = CalculerSecondLayer1092(&Layer2, output1);
//Dans cette config, La somme de LAYER 2 est OK
//     float *output3 = CalculerThirdLayer10(&Layer3, output2);

float sum2 = 0;

for (int i = 0; i < 1092; i++) {
    printf("output1[%d]: %.*f\n", i, DBL_DIG, output2[i]);
}
sum2 = sumVector(output2, 1092);

//     softmax(output3, 10);

printf("Sum of output2: %.*f\n",DBL_DIG, sum2);
// // for (int i = 0; i < 10; i++) {
// //     printf("output3[%d]: %.*f\n", i, DBL_DIG, output3[i]);
// // }

//     DesallouerBMP(&bitmap);
//     free(output1);
//     free(output2);
//     free(output3);


   return 0;
}