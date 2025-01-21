/**
 * \file main.c
 * \brief Example use of CNeural, finding Celsius to Fahrenheit function.
 *
 * \author Dai Duong Le
 * \version: 0.1.2
*/
#include <stdio.h>
#include <time.h>
#include "CNeural.h"

int main() {
    clock_t start = clock();

    NeuralNetwork ctof;

    int inputShape = 1;
    int outputShape = 1;
    int numLabels = 10; // same as features
    int numLayers = 4; // MUST be the same # of elements as in eachLayer[]
    int eachLayer[] = {4, 1000, 1000000, 1}; // should include output layer as well, the same as outputShape
    string afs[] = {"none", "none", "none", "none"}; // will not return 0 (error) when # of elements is < than # of layers, only checks for unknown af (strings)

    if (CNeural_init(&ctof, inputShape, outputShape, numLayers, eachLayer, afs, "random") != 0) {
        printf("Error: Initialization failed.");
        return 1;
    }

    // equation for calculating Celsius to Fahrenheit with 1 node
    // ctof.layers[0].nodes[0].weights[0] = (float) 1.8;
    // ctof.layers[0].nodes[0].bias = 32;

    float features[10][1] = {
        // {-273},
        {-40},
        {-10},
        {0},
        {8},
        {15},
        {22},
        {38},
        {45},
        {60},
        {80}
    };
    float labels[10][1] = {
        // {(float) -459.4},
        {-40},
        {14},
        {32},
        {46},
        {59},
        {72},
        {100},
        {113},
        {140},
        {176}
    };

    // printf("Weight: %f\n", ctof.layers[0].nodes[0].weights[0]);
    // printf("Bias: %f\n", ctof.layers[0].nodes[0].bias);

    CNeural_train(&ctof, numLabels, features, labels, "mse", "sgd", (float) 0.000005, 500, 100); // optimizer not implemented yet
    // printf("Weight: %f\n", ctof.layers[0].nodes[0].weights[0]);
    // printf("Bias: %f\n", ctof.layers[0].nodes[0].bias);
    // CNeural_free(&ctof);

    clock_t stop = clock();
    double elapsed = (double) (stop - start) / CLOCKS_PER_SEC;
    printf("\nTime elapsed: %.5fs\n", elapsed);

    return 0;
}


