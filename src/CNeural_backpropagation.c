/**
 * \file CNeural_backpropagation.c
 * \brief Source file for CNeural backpropagation, containing function definitions.
 *
 * \author Dai Duong Le
 * \version: 0.0.1
*/

#include "CNeural.h"
#include <string.h>
#include <math.h>
#include <stdio.h>

/**
 * Calculates the partial derivatives which combine to a gradient of all neural network parameters using backpropagation. 
 * 
 * @param nn neural network type
 * @param inputs array of inputs (features)
 * @param labels array of labels
 * @param lossFunction a string corresponding to a loss function
*/
void CNeural_derivatives(NeuralNetwork *nn, float inputs[], float labels[], string lossFunction) {
    for (int layerNum = nn->nLayers - 1; layerNum >= 0; layerNum--) {
        for (int nodeNum = 0; nodeNum < nn->layers[layerNum].nNodes; nodeNum++) {
            if (layerNum == 0) {   // first layer (last layer in backprop)
                for (int weightNum = 0; weightNum < nn->inShape; weightNum++) {

                }
            } else if (layerNum == nn->nLayers - 1) { // last layers
                for (int weightNum = 0; weightNum < nn->layers[layerNum - 1].nNodes; weightNum++) {
                    nn->layers[layerNum].nodes[nodeNum].weightDerivatives[weightNum] = nn->layers[layerNum-1].nodesResults[weightNum] * CNeural_af_derivative(nn->layers[layerNum].weightedSum[nodeNum], nn->layers[layerNum].nodes[nodeNum].AF) * CNeural_loss_derivative(nn->layers[layerNum].nodesResults[nodeNum], labels[nodeNum], lossFunction);
                }
                // bias
            } else { // middle layers

            }
        }
    }
}

/**
 * Calculates the partial derivative of the activation function with respect to a weighted sum. Helper function to CNeural_derivatives.
 *
 * @param input value of weighted sum
 * @param af a string corresponding to an activation function, values: "none", "sigmoid", "tanh", "relu", default: "relu"
 * @return rate of change of the activation function with respect to a weighted sum
*/
float CNeural_af_derivative(float input, string af) {
    if (strcmp(af, "none") == 0) return 1;
    if (strcmp(af, "sigmoid") == 0) {
        return 1 / (1 + expf(-input)) * (1 - 1 / (1 + expf(-input)));
    }
    if (strcmp(af, "tanh") == 0) {
        return 1 - powf(tanhf(input), 2);
    }
    if (strcmp(af, "relu") == 0) {
        if (input < 0) {
            return 0;
        }
        return 1;
    }
    printf("Warning: Unknown activation function. Training results might not be optimal!\n");
    printf("Defaulting to ReLu derivative\n");
    if (input < 0) {
        return 0;
    }
    return 1;
}

/**
 * Calculates the partial derivative of the loss function with respect to nodeResults. Helper function to CNeural_derivatives.
 *
 * @param predicted array of predicted values (nodeResult values of the last layer) for 1 training example
 * @param actual array of label values to compare to for 1 training example
 * @param lfn a string corresponding to a loss function, values: "mse", "mae", default: "mse"
 * @return rate of change of the loss function with respect to nodeResults
*/
float CNeural_loss_derivative(float predicted, float actual, string lfn) {
    if (strcmp(lfn, "mse") == 0) {
        return 2 * (predicted - actual);
    }
    if (strcmp(lfn, "mae") == 0) {
        // (might not be differentiable)
    }

    printf("Warning: Unknown loss function. Training results might not be optimal!\n");
    printf("Defaulting to MSE derivative\n");
    return 2 * (predicted - actual);
}