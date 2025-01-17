//
// Created by Dai Duong Le.
//

#include <stdio.h>
#include <time.h>
#include "CNeural.h"

int main() {
    clock_t start = clock();

    NeuralNetwork ctof;

    int inputShape = 1;
    int outputShape = 1;
    int numLabels = 3;
    int numLayers = 1; // MUST be the same # of elements as in eachLayer[]
    int eachLayer[] = {1}; // should include output layer as well, the same as outputShape
    string afs[] = {"none"}; // will not return 0 (error) when # of elements is < than # of layers, only checks for unknown af (strings)

    if (CNeural_init(&ctof, inputShape, outputShape, numLayers, eachLayer, afs, "random") != 0) {
        printf("Error: Initialization failed.");
        return 1;
    }

    // equation for calculating Celsius to Fahrenheit with 1 node
    ctof.layers[0].nodes[0].weights[0] = (float) 1.8;
    ctof.layers[0].nodes[0].bias = 32;

    float features[3][1] = {
        {-273},
        {0},
        {21},
        // {100}
    };
    float labels[3][1] = {
        {(float) -459.4},
        {32},
        {(float) 69.8},
        // {212}
    };

    CNeural_train(&ctof, numLabels, features, labels, "mse", "sgd", (float) 0.1, 100);
    CNeural_free(&ctof);

    clock_t stop = clock();
    double elapsed = (double) (stop - start) / CLOCKS_PER_SEC;
    printf("\nTime elapsed: %.5f\n", elapsed);

    return 0;
}


