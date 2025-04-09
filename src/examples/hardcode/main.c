/**
 * \file main.c
 * \brief Example use of CNeural, finding Celsius to Fahrenheit function.
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

    NeuralNetwork *ctof = malloc( sizeof(NeuralNetwork) );
    if (ctof == NULL) {
        printf("malloc failed\n");
    }

    int inputShape = 2;
    int outputShape = 3;
    int numLabels = 3; // same as features
    int numLayers = 2; // MUST be the same # of elements as in eachLayer[]
    int eachLayer[] = {2, 3}; // should include output layer as well, the same as outputShape
    string afs[] = {"relu", "softmax"}; // will not return 0 (error) when # of elements is < than # of layers, only checks for unknown af (strings)

    if (CNeural_init(ctof, inputShape, outputShape, numLayers, eachLayer, afs, "random") != 0) {
        printf("Error: Initialization failed.");
        return 1;
    }
    printf("Initialization successful.\n");
    // printf("Test: %d", ctof.nLayers);
    // equation for calculating Celsius to Fahrenheit with 1 node
    ctof->layers[0].nodes[0].weights[0] = (float) -2.5;
    ctof->layers[0].nodes[0].weights[1] = (float) 0.6;
    ctof->layers[0].nodes[0].bias = (float) 1.6;

    ctof->layers[0].nodes[1].weights[0] = (float) -1.5;
    ctof->layers[0].nodes[1].weights[1] = (float) 0.4;
    ctof->layers[0].nodes[1].bias = (float) 0.7;
    //

    ctof->layers[1].nodes[0].weights[0] = (float) -0.1;
    ctof->layers[1].nodes[0].weights[1] = (float) 1.5;
    ctof->layers[1].nodes[0].bias = (float) -2;

    ctof->layers[1].nodes[1].weights[0] = (float) 2.4;
    ctof->layers[1].nodes[1].weights[1] = (float) -5.2;
    ctof->layers[1].nodes[1].bias = (float) 0;

    ctof->layers[1].nodes[2].weights[0] = (float) -2.2;
    ctof->layers[1].nodes[2].weights[1] = (float) 3.7;
    ctof->layers[1].nodes[2].bias = (float) 1;

    float features[3][2] = {
        // {-273},
        {0.04, 0.42},
        {1, 0.54},
        {0.5, 0.37}
    };
    float labels[3][3] = {
        // {(float) -459.4},
        {1, 0, 0},
        {0, 1, 0},
        {0, 0, 1}

    };

    // printf("Weight: %f\n", ctof.layers[0].nodes[0].weights[0]);
    // printf("Bias: %f\n", ctof.layers[0].nodes[0].bias);
    FILE* lossFile = fopen("loss.csv", "a");
    if (lossFile == NULL) {
        printf("Error opening loss file\n");
    }
    // printf("Weight: %f\n", ctof.layers[0].nodes[0].weights[0]);
    // printf("Bias: %f\n", ctof.layers[0].nodes[0].bias);
    CNeural_train(ctof, numLabels, features, labels, "categorical_cross_entropy", "sgd", (float) 1, 3, 0, lossFile); // optimizer not implemented yet
    // printf("Weight: %f\n", ctof.layers[0].nodes[0].weights[0]);
    // printf("Bias: %f\n", ctof.layers[0].nodes[0].bias);

    printf("Bias: %f\n", ctof->layers[1].nodes[0].bias);
    // printf("prediction: ");
    CNeural_predict(ctof, features[0]);
    printf("\n");
    CNeural_predict(ctof, features[1]);
    printf("\n");
    CNeural_predict(ctof, features[0]);
    printf("\n");
    CNeural_predict(ctof, features[2]);
    // CNeural_free(&ctof);

    clock_t stop = clock();
    double elapsed = (double) (stop - start) / CLOCKS_PER_SEC;
    printf("\nTime elapsed: %.5fs\n", elapsed);

    return 0;
}


