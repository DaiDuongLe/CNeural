/**
 * \file main.c
 * \brief Example use of CNeural, classifying Iris species using petal width and length.
 *
 * \author Dai Duong Le
 * \version: 1.0.0
*/
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "../../CNeural.h"

int main() {
    clock_t start = clock();

    NeuralNetwork *iris = malloc( sizeof(NeuralNetwork) );
    if (iris == NULL) {
        printf("malloc failed\n");
        return 1;
    }

    int inputShape = 2;
    int outputShape = 3;
    int numLabels = 150; // same as features
    int numLayers = 2; // MUST be the same # of elements as in eachLayer[]
    int eachLayer[] = {6, 3}; // should include output layer as well, the same as outputShape
    string afs[] = {"relu", "softmax"}; // will not return 0 (error) when # of elements is < than # of layers, only checks for unknown af (strings)

    if (CNeural_init(iris, inputShape, outputShape, numLayers, eachLayer, afs, "random") != 0) {
        printf("Error: Initialization failed.");
        return 1;
    }
    printf("Initialization successful.\n");

    float features[150][2] = {
        {1.4, 0.2}, {1.4, 0.2}, {1.3, 0.2}, {1.5, 0.2}, {1.4, 0.2},
        {1.7, 0.4}, {1.4, 0.3}, {1.5, 0.2}, {1.4, 0.2}, {1.5, 0.1},
        {1.5, 0.2}, {1.6, 0.2}, {1.4, 0.1}, {1.1, 0.1}, {1.2, 0.2},
        {1.5, 0.4}, {1.3, 0.4}, {1.4, 0.3}, {1.7, 0.3}, {1.5, 0.3},
        {1.7, 0.2}, {1.5, 0.4}, {1.0, 0.2}, {1.7, 0.5}, {1.9, 0.2},
        {1.6, 0.2}, {1.6, 0.4}, {1.5, 0.2}, {1.4, 0.2}, {1.6, 0.2},
        {1.6, 0.2}, {1.5, 0.4}, {1.5, 0.1}, {1.4, 0.2}, {1.5, 0.1},
        {1.2, 0.2}, {1.3, 0.2}, {1.5, 0.1}, {1.3, 0.2}, {1.5, 0.2},
        {1.3, 0.3}, {1.3, 0.3}, {1.3, 0.2}, {1.6, 0.6}, {1.9, 0.4},
        {1.4, 0.3}, {1.6, 0.2}, {1.4, 0.2}, {1.5, 0.2}, {1.4, 0.2},

        {4.7, 1.4}, {4.5, 1.5}, {4.9, 1.5}, {4.0, 1.3}, {4.6, 1.5},
        {4.5, 1.3}, {4.7, 1.6}, {3.3, 1.0}, {4.6, 1.3}, {3.9, 1.4},
        {3.5, 1.0}, {4.2, 1.5}, {4.0, 1.0}, {4.7, 1.4}, {3.6, 1.3},
        {4.4, 1.4}, {4.5, 1.5}, {4.1, 1.0}, {4.5, 1.5}, {3.9, 1.1},
        {4.8, 1.8}, {4.0, 1.3}, {4.9, 1.5}, {4.7, 1.2}, {4.3, 1.3},
        {4.4, 1.4}, {4.8, 1.4}, {5.0, 1.7}, {4.5, 1.5}, {3.5, 1.0},
        {3.8, 1.1}, {3.7, 1.0}, {3.9, 1.2}, {5.1, 1.6}, {4.5, 1.5},
        {4.5, 1.6}, {4.7, 1.5}, {4.4, 1.3}, {4.1, 1.3}, {4.0, 1.3},
        {4.4, 1.2}, {4.6, 1.4}, {4.0, 1.2}, {3.3, 1.0}, {4.2, 1.3},
        {4.2, 1.2}, {4.2, 1.3}, {4.3, 1.3}, {3.0, 1.1}, {4.1, 1.3},

        {6.0, 2.5}, {5.1, 1.9}, {5.9, 2.1}, {5.6, 1.8}, {5.8, 2.2},
        {6.6, 2.1}, {4.5, 1.7}, {6.3, 1.8}, {5.8, 1.8}, {6.1, 2.5},
        {5.1, 2.0}, {5.3, 1.9}, {5.5, 2.1}, {5.0, 2.0}, {5.1, 2.4},
        {5.3, 2.3}, {5.5, 1.8}, {6.7, 2.2}, {6.9, 2.3}, {5.0, 1.5},
        {5.7, 2.3}, {4.9, 2.0}, {6.7, 2.0}, {4.9, 1.8}, {5.7, 2.1},
        {6.0, 1.8}, {4.8, 1.8}, {4.9, 1.8}, {5.6, 2.1}, {5.8, 1.6},
        {6.1, 1.9}, {6.4, 2.0}, {5.6, 2.2}, {5.1, 1.5}, {5.6, 1.4},
        {6.1, 2.3}, {5.6, 2.4}, {5.5, 1.8}, {4.8, 1.8}, {5.4, 2.1},
        {5.6, 2.4}, {5.1, 2.3}, {5.1, 1.9}, {5.9, 2.3}, {5.7, 2.5},
        {5.2, 2.3}, {5.0, 1.9}, {5.2, 2.0}, {5.4, 2.3}, {5.1, 1.8}
    };
    float labels[150][3];

    for (int i = 0; i < numLabels; i++) {
        if (i < 50) {             // Iris-setosa
            labels[i][0] = 1; labels[i][1] = 0; labels[i][2] = 0;
        } else if (i < 100) {     // Iris-versicolor
            labels[i][0] = 0; labels[i][1] = 1; labels[i][2] = 0;
        } else {                  // Iris-virginica
            labels[i][0] = 0; labels[i][1] = 0; labels[i][2] = 1;
        }
    }

    FILE* lossFile = fopen("loss.csv", "a");
    if (lossFile == NULL) {
        printf("Error opening loss file\n");
    }

    CNeural_train(iris, numLabels, features, labels, "categorical_cross_entropy", "sgd", (float) 0.001, 250, 30, lossFile); // optimizer not implemented yet

    printf("Setosa: \n");
    CNeural_predict(iris, features[0]);
    printf("\n");
    CNeural_predict(iris, features[15]);
    printf("\n");
    CNeural_predict(iris, features[39]);
    printf("\n");
    CNeural_predict(iris, features[46]);
    printf("\n");

    printf("Versicolor: \n");
    CNeural_predict(iris, features[55]);
    printf("\n");
    CNeural_predict(iris, features[67]);
    printf("\n");
    CNeural_predict(iris, features[88]);
    printf("\n");
    CNeural_predict(iris, features[90]);
    printf("\n");

    printf("Virginica: \n");
    CNeural_predict(iris, features[110]);
    printf("\n");
    CNeural_predict(iris, features[120]);
    printf("\n");
    CNeural_predict(iris, features[135]);
    printf("\n");
    CNeural_predict(iris, features[144]);
    CNeural_free(iris);

    clock_t stop = clock();
    double elapsed = (double) (stop - start) / CLOCKS_PER_SEC;
    printf("\nTime elapsed: %.5fs\n", elapsed);

    return 0;
}


