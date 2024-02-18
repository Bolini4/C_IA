#include <stdio.h>
#include <stdlib.h>
#include <float.h>

#include "Bmp2Matrix.h"
#include "functions.h"

int main(int argc, char* argv[]){

    DenseLayer Layer1, Layer2, Layer3;

    printf("Loading weights and biases...\n");
    //loading of layer 1 OK
    loadWeightsAndBiases(&Layer1, "./weightandbiases/layer_1_weights.txt", "./weightandbiases/layer_1_biases.txt", 64, 784);
    loadWeightsAndBiases(&Layer2, "./weightandbiases/layer_2_weights.txt", "./weightandbiases/layer_2_biases.txt", 1092, 64);
    loadWeightsAndBiases(&Layer3, "./weightandbiases/layer_3_weights.txt", "./weightandbiases/layer_3_biases.txt", 10, 1092);
    printf("Weights and biases loaded successfully\n");



double test = 0.123456789;
printf("%.*f\n",DBL_DIG, test);

printf("Loading image...fdjsklfdskjfldsjfkdlsfdlksj\n");

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

// for (int i = 0; i < 784; i++) {
//     printf("%.*f\n",DBL_DIG, flatImage[i]);
// }

    float *output1 = CalculerFirstLayer64(&Layer1, flatImage);
    printf("%f\n", Layer1.weights[783][63]);
    //OUTPUT OF LAYER 1 IS LIKE IN PYTHON
    float *output2 = CalculerSecondLayer1092(&Layer2, output1);
    //second layer looks to be OK but there is some problem d'arrondis...


    float *output3 = CalculerThirdLayer10(&Layer3, output2);

    printf("Output of layer 1: \n");

    softmax(output3, 10);
for (int i = 0; i < 10; i++) {
    printf("%.*f\n",DBL_DIG, output3[i]);
    
}

// printf("%f\n", output1[100]);

    DesallouerBMP(&bitmap);
    free(output1);
    free(output2);
    free(output3);


    
   return 0;
}