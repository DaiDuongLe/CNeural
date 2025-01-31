#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "../../CNeural.h"

int main() {
    clock_t start = clock();
    FILE *trainingImageFile = fopen("C:/Users/ledai/CLionProjects/CNeural/data/mnist/train-images.idx3-ubyte", "r");
    FILE *trainingLabelFile = fopen("C:/Users/ledai/CLionProjects/CNeural/data/mnist/train-labels.idx1-ubyte", "r");

    if (trainingImageFile == NULL || trainingLabelFile == NULL) {
        printf("Error opening files\n");
    }

    char* trainingDataset = malloc(47040016);
    fread(trainingDataset, sizeof(char), 47040016, trainingImageFile);
    char* trainingLabels = malloc(60008);
    fread(trainingLabels, sizeof(char), 60008, trainingLabelFile);

    // normalize training data from 0 to 255 into 0 to 1
    for (int i = 16; i < 47040016; i++) {
      if (trainingDataset[i] != 0) {
        trainingDataset[i] = 1;
      }
    }

    char* trainArr[59997];
    for (int i = 0; i < 59998; i++) {
        trainArr[i] = trainingDataset + 16 + i * 784;
    }
//    char* labelCharArr[59997];
//    for (int i = 0; i < 59998; i++) {
//      labelCharArr[i] = trainingLabels + 8 + i;
//    }
    float* labelArr[59997];

    float arr0[] = {1, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    float arr1[] = {0, 1, 0, 0, 0, 0, 0, 0, 0, 0};
    float arr2[] = {0, 0, 1, 0, 0, 0, 0, 0, 0, 0};
    float arr3[] = {0, 0, 0, 1, 0, 0, 0, 0, 0, 0};
    float arr4[] = {0, 0, 0, 0, 1, 0, 0, 0, 0, 0};
    float arr5[] = {0, 0, 0, 0, 0, 1, 0, 0, 0, 0};
    float arr6[] = {0, 0, 0, 0, 0, 0, 1, 0, 0, 0};
    float arr7[] = {0, 0, 0, 0, 0, 0, 0, 1, 0, 0};
    float arr8[] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 0};
    float arr9[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 1};

    for (int i = 0; i < 59998; i++) {
      switch (*(trainingLabels + 8 + i)) {
        case 0:
          labelArr[i] = arr0;
          break;
        case 1:
          labelArr[i] = arr1;
          break;
        case 2:
          labelArr[i] = arr2;
          break;
         case 3:
          labelArr[i] = arr3;
          break;
         case 4:
          labelArr[i] = arr4;
          break;
         case 5:
          labelArr[i] = arr5;
          break;
         case 6:
          labelArr[i] = arr6;
          break;
         case 7:
          labelArr[i] = arr7;
          break;
         case 8:
          labelArr[i] = arr8;
          break;
         case 9:
           labelArr[i] = arr9;
           break;
      }
    }

//    printf("Training label: %d\n", *(trainingLabels + 8 + 1253));
//    for (int i = 0; i < 10; i++) {
//      printf("%f ", labelArr[1253][i]);
//    }

    NeuralNetwork mnist;

    int inputShape = 784;
    int outputShape = 10;
    int numLabels = 59997; // same as features
    int numLayers = 3; // MUST be the same # of elements as in eachLayer[]
    int eachLayer[] = {16, 16, 10}; // should include output layer as well, the same as outputShape
    string afs[] = {"sigmoid", "sigmoid", "sigmoid"}; // will not return 0 (error) when # of elements is < than # of layers, only checks for unknown af (strings)

    if (CNeural_init(&mnist, inputShape, outputShape, numLayers, eachLayer, afs, "random") != 0) {
      printf("Error: Initialization failed.");
      return 1;
    }
    CNeural_train_ptr(&mnist, numLabels, trainArr, labelArr, "mse", "sgd", (float) 0.5, 50, 1); // optimizer not implemented yet

    // display image
    int imageNumber = 420;
    for (int i = 0; i < 28; i++) {
        for (int j = 0; j < 27; j++) {
            if (trainArr[imageNumber - 1][i * 28 + j] == 0) {
                printf("%d", trainArr[imageNumber - 1][i * 28 + j]);
            } else if (trainArr[imageNumber - 1][i * 28 + j] == 1) {
                printf("%d", trainArr[imageNumber - 1][i * 28 + j]);
            }
        }
        printf("\n");
    }
    CNeural_predict_ptr(&mnist, trainArr[imageNumber - 1]);
//    CNeural_free(&mnist);

    clock_t stop = clock();
    double elapsed = (double) (stop - start) / CLOCKS_PER_SEC;
    printf("\nTime elapsed: %.5fs\n", elapsed);

    return 0;
//    for (int i = 0; i < 28; i++) {
//      for (int j = 0; j < 27; j++) {
//        if (trainingDataset[16 + i * 28 + j + (59997 * 784)] == 0) {
//          printf("-");
//        } else {
//          printf("+");
//        }
//      }
//      printf("\n");
//    }

//    for (int j = 0; j < 60000; j = j + 784) {
//        for (int k = 0; k < 784; k++) {
////          printf("%d", trainingDataset[16 + j + k]);
//            if (trainingDataset[16 + j + k] == 0) {
//              printf(".");
//            } else {
//              printf("#");
//            }
//        }
//          printf("\n");
//    }

//    while ((dataRead = fgetc(trainingImageFile)) != EOF) {
//      printf("%i", dataRead);
//    }
//    while (ftell(trainingImageFile) < (long) 4) {
//        printf("Current file position: %ld\n", ftell(trainingImageFile));
////        fread(&dataRead, 1, 1, trainingImageFile);
////        int dataRead = fgetc(trainingImageFile);
////        printf("%d\n", dataRead);
//        fseek(trainingImageFile, 1, SEEK_CUR);
//    }
}