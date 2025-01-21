/**
 * \file CNeural_backpropagation.c
 * \brief Source file for CNeural backpropagation, containing function definitions.
 *
 * \author Dai Duong Le
 * \version: 0.1.2
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
            if (layerNum == 0) { // last layer in backprop (first layer)
                if (nn->nLayers == 1) { // if there's only 1 layer (no hidden layers)
                    for (int weightNum = 0; weightNum < nn->inShape; weightNum++) {
                        float previousDerivative = nn->layers[layerNum].nodes[nodeNum].weightDerivatives[weightNum];
                        // printf("WeightderBEFORE: %f\n", previousDerivative);
                        float currentDerivative = (float) 1.0/(float) nn->nLabels * inputs[weightNum] * CNeural_af_derivative(nn->layers[layerNum].weightedSum[nodeNum], nn->layers[layerNum].nodes[nodeNum].AF) * CNeural_loss_derivative(nn->layers[layerNum].nodesResults[nodeNum], labels[nodeNum], lossFunction);
                        // printf("WeightderCURRENT: %f\n", currentDerivative);
                        nn->layers[layerNum].nodes[nodeNum].weightDerivatives[weightNum] = previousDerivative + currentDerivative;
                    }
                    float previousDerivative = nn->layers[layerNum].nodes[nodeNum].biasDerivative;
                    // printf("BiasderPREVIOUS: %f\n", previousDerivative);
                    float currentDerivative = (float) 1.0/(float) nn->nLabels * CNeural_af_derivative(nn->layers[layerNum].weightedSum[nodeNum], nn->layers[layerNum].nodes[nodeNum].AF) * CNeural_loss_derivative(nn->layers[layerNum].nodesResults[nodeNum], labels[nodeNum], lossFunction);
                    // printf("BiasderCURRENT: %f\n", currentDerivative);
                    nn->layers[layerNum].nodes[nodeNum].biasDerivative = previousDerivative + currentDerivative;
                } else {
                    float nodeResultDerivativeSum = 0; // partial derivative of the cost with respect to an element in the current layer nodesResults (NodeResultDerivative)
                    for (int i = 0; i < nn->layers[layerNum + 1].nNodes; i++) {
                        float previousWeightNodeResultDerivative = nn->layers[layerNum + 1].nodes[i].weights[nodeNum] * nn->layers[layerNum + 1].nodesResultsDerivatives[i]; // current weight is nodes[i].weights[nodeNum]
                        nodeResultDerivativeSum += previousWeightNodeResultDerivative;
                    }
                    nn->layers[layerNum].nodesResultsDerivatives[nodeNum] = nodeResultDerivativeSum;

                    for (int weightNum = 0; weightNum < nn->inShape; weightNum++) {
                        float previousDerivative = nn->layers[layerNum].nodes[nodeNum].weightDerivatives[weightNum];
                        // printf("WeightderBEFORE: %f\n", previousDerivative);
                        float currentDerivative = (float) 1.0/(float) nn->nLabels * inputs[weightNum] * CNeural_af_derivative(nn->layers[layerNum].weightedSum[nodeNum], nn->layers[layerNum].nodes[nodeNum].AF) * nodeResultDerivativeSum;
                        // printf("WeightderCURRENT: %f\n", currentDerivative);
                        nn->layers[layerNum].nodes[nodeNum].weightDerivatives[weightNum] = previousDerivative + currentDerivative;
                    }
                    float previousDerivative = nn->layers[layerNum].nodes[nodeNum].biasDerivative;
                    // printf("BiasderPREVIOUS: %f\n", previousDerivative);
                    float currentDerivative = (float) 1.0/(float) nn->nLabels * CNeural_af_derivative(nn->layers[layerNum].weightedSum[nodeNum], nn->layers[layerNum].nodes[nodeNum].AF) * nodeResultDerivativeSum;
                    // printf("BiasderCURRENT: %f\n", currentDerivative);
                    nn->layers[layerNum].nodes[nodeNum].biasDerivative = previousDerivative + currentDerivative;
                }
            } else if (layerNum == nn->nLayers - 1) { // first layer in backprop (last layer)
                nn->layers[layerNum].nodesResultsDerivatives[nodeNum] = CNeural_af_derivative(nn->layers[layerNum].weightedSum[nodeNum], nn->layers[layerNum].nodes[nodeNum].AF) * CNeural_loss_derivative(nn->layers[layerNum].nodesResults[nodeNum], labels[nodeNum], lossFunction);
                for (int weightNum = 0; weightNum < nn->layers[layerNum - 1].nNodes; weightNum++) {
                    float previousDerivative = nn->layers[layerNum].nodes[nodeNum].weightDerivatives[weightNum];
                    // printf("WeightderBEFORE: %f\n", previousDerivative);
                    float currentDerivative = (float) 1.0/(float) nn->nLabels * nn->layers[layerNum - 1].nodesResults[weightNum] * nn->layers[layerNum].nodesResultsDerivatives[nodeNum];
                    // printf("WeightderCURRENT: %f\n", currentDerivative);
                    nn->layers[layerNum].nodes[nodeNum].weightDerivatives[weightNum] = previousDerivative + currentDerivative;
                }
                float previousDerivative = nn->layers[layerNum].nodes[nodeNum].biasDerivative;
                // printf("BiasderPREVIOUS: %f\n", previousDerivative);
                float currentDerivative = (float) 1.0/(float) nn->nLabels * nn->layers[layerNum].nodesResultsDerivatives[nodeNum];
                // printf("BiasderCURRENT: %f\n", currentDerivative);
                nn->layers[layerNum].nodes[nodeNum].biasDerivative = previousDerivative + currentDerivative;
            } else { // hidden layers except for first (last in backprop)
                float nodeResultDerivativeSum = 0; // partial derivative of the cost with respect to an element in the current layer nodesResults (NodeResultDerivative)
                for (int i = 0; i < nn->layers[layerNum + 1].nNodes; i++) {
                    float previousWeightNodeResultDerivative = nn->layers[layerNum + 1].nodes[i].weights[nodeNum] * nn->layers[layerNum + 1].nodesResultsDerivatives[i]; // current weight is nodes[i].weights[nodeNum]
                    nodeResultDerivativeSum += previousWeightNodeResultDerivative;
                }
                nn->layers[layerNum].nodesResultsDerivatives[nodeNum] = nodeResultDerivativeSum;

                for (int weightNum = 0; weightNum < nn->layers[layerNum - 1].nNodes; weightNum++) {
                    float previousDerivative = nn->layers[layerNum].nodes[nodeNum].weightDerivatives[weightNum];
                    // printf("WeightderBEFORE: %f\n", previousDerivative);
                    float currentDerivative = (float) 1.0/(float) nn->nLabels * nn->layers[layerNum - 1].nodesResults[weightNum] * CNeural_af_derivative(nn->layers[layerNum].weightedSum[nodeNum], nn->layers[layerNum].nodes[nodeNum].AF) * nodeResultDerivativeSum;
                    // printf("WeightderCURRENT: %f\n", currentDerivative);
                    nn->layers[layerNum].nodes[nodeNum].weightDerivatives[weightNum] = previousDerivative + currentDerivative;
                }
                float previousDerivative = nn->layers[layerNum].nodes[nodeNum].biasDerivative;
                // printf("BiasderPREVIOUS: %f\n", previousDerivative);
                float currentDerivative = (float) 1.0/(float) nn->nLabels * CNeural_af_derivative(nn->layers[layerNum].weightedSum[nodeNum], nn->layers[layerNum].nodes[nodeNum].AF) * nodeResultDerivativeSum;
                // printf("BiasderCURRENT: %f\n", currentDerivative);
                nn->layers[layerNum].nodes[nodeNum].biasDerivative = previousDerivative + currentDerivative;
            }
        }
    }
}

/**
 * Updates weights and biases based on the calculated gradient and applies its negative.
 *
 * @param nn neural network type
*/
void CNeural_update_weights(NeuralNetwork *nn) {
    for (int layerNum = 0; layerNum < nn->nLayers; layerNum++) {
        for (int nodeNum = 0; nodeNum < nn->layers[layerNum].nNodes; nodeNum++) {
            if (layerNum == 0) { // 1st layer # of weights should = # of inputs
                for (int weightNum = 0; weightNum < nn->inShape; weightNum++) {
                    nn->layers[layerNum].nodes[nodeNum].weights[weightNum] += nn->lr * -nn->layers[layerNum].nodes[nodeNum].weightDerivatives[weightNum]; // negative gradient (downhill direction)
                }
            } else {  // # of weights should = previous layer # of nodes
                for (int weightNum = 0; weightNum < nn->layers[layerNum - 1].nNodes; weightNum++) {
                    nn->layers[layerNum].nodes[nodeNum].weights[weightNum] += nn->lr * -nn->layers[layerNum].nodes[nodeNum].weightDerivatives[weightNum]; // negative gradient (downhill direction)
                }
            }
            nn->layers[layerNum].nodes[nodeNum].bias += nn->lr * -nn->layers[layerNum].nodes[nodeNum].biasDerivative; // negative gradient (downhill direction)
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