/*******************************************************
Nom ......... : main.c
Role ........ : Programme principal executant la lecture
                d'une image bitmap
Auteur ...... : Frédéric CHATRIE
Version ..... : V1.1 du 1/2/2021
Licence ..... : /

Compilation :
make veryclean
make
Pour exécuter, tapez : ./all
********************************************************/

#include <stdio.h>
#include <stdlib.h>

#include "Bmp2Matrix.h"

int main(int argc, char* argv[]){
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
   for (int i=0; i<28; i++)
    for (int j=0; j<28; j++)
    printf("%d\n", bitmap.mPixelsGray[i][j]);
   DesallouerBMP(&bitmap);

   return 0;
}
