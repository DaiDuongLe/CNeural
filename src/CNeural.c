/**
 * \file CNeural.c
 * \brief Source file for CNeural, containing function definitions.
 *
 * \author Dai Duong Le
 * \version: 0.1.1
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include "CNeural.h"

#define AFStringSize 10 // TODO malloc for AF strings

/* TODO WeightBias Proper Init, add better init options (Xavier, etc.)
// TODO AF strings not allocated
// TODO Make strings enums?
// TODO Labeling Issue? - classification & Prediction function
// TODO Data Import - MNIST
// TODO Save/Import Weights
// TODO Check input/return value validation
// TODO (optional) better interface - gui?
*/

/**
 * Initializes a neural network.
 *
 * @param nn neural network type
 * @param inputShape input number of nodes
 * @param outputShape output number of nodes
 * @param numLayers number of layers
 * @param layerNumNodes array with each layer's number of nodes
 * @param layersAF array with each layer's activation function
 * @param initMethod initialization method
 * @return returns 0 for success
*/
int CNeural_init(NeuralNetwork *nn, int inputShape, int outputShape, int numLayers, int layerNumNodes[], string layersAF[], string initMethod) {
    nn->inShape = inputShape;
    nn->outShape = outputShape;
    nn->nLayers = numLayers;

    nn->layers = malloc(sizeof(Layer) * (unsigned int) nn->nLayers);
    if (nn->layers == NULL) {
        printf("Error: Failed to allocate memory!");
        return 1;
    }

    for (int i = 0; i < nn->nLayers; i++) { // initialize each LAYER with nodes and activation functions
        nn->layers[i].nNodes = layerNumNodes[i];
        nn->layers[i].layerAF = layersAF[i];


        nn->layers[i].nodes = malloc(sizeof(Node) * (unsigned int) nn->layers[i].nNodes);
        nn->layers[i].weightedSum = malloc(sizeof(float) * (unsigned int) nn->layers[i].nNodes);
        nn->layers[i].nodesResults = malloc(sizeof(float) * (unsigned int) nn->layers[i].nNodes);
        nn->layers[i].nodesResultsDerivatives = malloc(sizeof(float) * (unsigned int) nn->layers[i].nNodes);
        CNeural_clear_nodeResults(nn, i); // init with 0 to clear garbage values

        if (nn->layers[i].nodes == NULL || nn->layers[i].weightedSum == NULL || nn->layers[i].nodesResults == NULL) {
            printf("Error: Failed to allocate memory!");
            return 1;
        }

        if (CNeural_wb_init(nn, i, initMethod) != 0) {
            return 1;
        }
    }

    return 0;
}

/**
 * Initializes the weights and biases. Helper function to CNeural_init.
 *
 * @param nn neural network type
 * @param layerNum layer number
 * @param option a string corresponding to an initialization method, values: "zero", "random"
 * @return returns 0 for sucess
*/
int CNeural_wb_init(NeuralNetwork *nn, int layerNum, string option) {
    //TODO (optional) calculate mean, variance, standard deviation for measuring appropriateness
    srand((unsigned int) time(NULL)); // init random generator

    if (strcmp(option, "zero") == 0) {
        // probably not a good init option
    } else if (strcmp(option, "random") == 0) {
        for (int i = 0; i < nn->layers[layerNum].nNodes; i++) { // for each NODE in layer init weights
            if (layerNum == 0) { // first layer # of weights should = # of inputs
                nn->layers[layerNum].nodes[i].weights = malloc(sizeof(float) * (unsigned int) nn->inShape);
                nn->layers[layerNum].nodes[i].weightDerivatives = malloc(sizeof(float) * (unsigned int) nn->inShape);
                if (nn->layers[layerNum].nodes[i].weights == NULL || nn->layers[layerNum].nodes[i].weightDerivatives == NULL) { printf("Error: Failed to allocate memory!"); return 1; }
                for (int j = 0; j < nn->inShape; j++) { // for each WEIGHT in node
                    nn->layers[layerNum].nodes[i].weights[j] = (float) (rand() / ((double) RAND_MAX + 1.0));
                    nn->layers[layerNum].nodes[i].weightDerivatives[j] = 0;
                }
            } else {  // # of weights should = previous layer # of nodes
                nn->layers[layerNum].nodes[i].weights = malloc(sizeof(float) * (unsigned int) nn->layers[layerNum - 1].nNodes);
                nn->layers[layerNum].nodes[i].weightDerivatives = malloc(sizeof(float) * (unsigned int) nn->layers[layerNum - 1].nNodes);
                if (nn->layers[layerNum].nodes[i].weights == NULL || nn->layers[layerNum].nodes[i].weightDerivatives == NULL) { printf("Error: Failed to allocate memory!"); return 1; }
                for (int j = 0; j < nn->layers[layerNum - 1].nNodes; j++) { // for each WEIGHT in node
                    nn->layers[layerNum].nodes[i].weights[j] = (float) (rand() / ((double) RAND_MAX + 1.0));
                    nn->layers[layerNum].nodes[i].weightDerivatives[j] = 0;
                }
            }
            nn->layers[layerNum].nodes[i].bias = 0; // bias can be 0
            nn->layers[layerNum].nodes[i].biasDerivative = 0;
            nn->layers[layerNum].nodes[i].AF = nn->layers[layerNum].layerAF; // applies to the whole layer
        }
    } else {
        printf("Error: Unknown initialization method.");
        return 1;
    }

    return 0;
}

/**
 * Clears nodeResults array.
 *
 * @param nn neural network type
 * @param layerNum layer number
*/
void CNeural_clear_nodeResults(NeuralNetwork *nn, int layerNum) { // TODO Rename to clear_layerParams or similar
    for (int i = 0; i < nn->layers[layerNum].nNodes; i++) {
        nn->layers[layerNum].weightedSum[i] = 0; // and weightedSum
        nn->layers[layerNum].nodesResults[i] = 0;
        nn->layers[layerNum].nodesResultsDerivatives[i] = 0;
    }
}

/**
 * Trains a neural network using the specified parameters.
 *
 * Weighted sums get forward propagated (forward pass) by calculating linear combinations and applying an activation function for each node in each layer.
 * After all the training examples go through 1 epoch, the gradient and loss are calculated based on the optimizer.
 * Neural network parameters are adjusted with the learning rate accordingly.
 *
 * @param nn neural network type
 * @param numLabels number of labels (training examples)
 * @param inputs 2D array of inputs (features)
 * @param labels 2D array of labels
 * @param lossFunction a string corresponding to a loss function
 * @param optimizer a string corresponding to an optimizer
 * @param learningRate learning rate to be applied in gradient descent
 * @param epochs number of epochs (forward passes through the whole dataset)
 * @param earlyStopLoss stop at or below specified loss value
*/
void CNeural_train(NeuralNetwork *nn, int numLabels, float inputs[numLabels][nn->inShape], float labels[numLabels][nn->outShape], string lossFunction, string optimizer, float learningRate, int epochs, float earlyStopLoss) {
    nn->nLabels = numLabels;
    nn->lf = lossFunction;
    nn->opt = optimizer;
    nn->lr = learningRate;
    nn->epochs = epochs;

    // TODO loss and expand activation functions
    for (int epoch = 1; epoch <= nn->epochs; epoch++) {
        printf("Epoch %d/%d\n", epoch, nn->epochs);
        for (int label = 0; label < numLabels; label++) {
            // printf("Label: %d\n", label + 1);

            for (int layerNum = 0; layerNum < nn->nLayers; layerNum++) {
                // printf("Layer %d\n", layerNum + 1);

                for (int nodeNum = 0; nodeNum < nn->layers[layerNum].nNodes; nodeNum++) {
                    // printf("\tNode %d\n", nodeNum + 1);
                    if (layerNum == 0) { // 1st layer # of weights should = # of inputs
                        for (int weightNum = 0; weightNum < nn->inShape; weightNum++) {
                            // printf("\t\tWeight %d: %f ", weightNum + 1, nn->layers[0].nodes[nodeNum].weights[weightNum]);
                            // printf("\t\tWeightder %d: %f ", weightNum + 1, nn->layers[0].nodes[nodeNum].weightDerivatives[weightNum]);
                            nn->layers[layerNum].nodesResults[nodeNum] +=
                                nn->layers[layerNum].nodes[nodeNum].weights[weightNum] * inputs[label][weightNum]; // adds for each linear combination (weighted sum)

                            // printf("Noderes value: %f\n", nn->layers[0].nodesResults[nodeNum]);
                        }
                    } else {  // # of weights should = previous layer # of nodes
                        for (int weightNum = 0; weightNum < nn->layers[layerNum - 1].nNodes; weightNum++) {
                            // printf("\t\tWeight %d: %f \t", weightNum + 1, nn->layers[layerNum].nodes[nodeNum].weights[weightNum]);
                            nn->layers[layerNum].nodesResults[nodeNum] +=
                                nn->layers[layerNum].nodes[nodeNum].weights[weightNum] * nn->layers[layerNum - 1].nodesResults[weightNum]; // adds for each linear combination (weighted sum)
                            // printf("Noderes value: %f\n", nn->layers[layerNum].nodesResults[nodeNum]);
                        }
                    }
                    // printf("\t\tBiasder: %f ", nn->layers[0].nodes[nodeNum].biasDerivative);
                    nn->layers[layerNum].nodesResults[nodeNum] += nn->layers[layerNum].nodes[nodeNum].bias;
                    // printf("\n");
                    // printf("\t\tAfter bias: %f\n", nn->layers[layerNum].nodesResults[nodeNum]);

                    // printf("\t\tAfter activation: %f\n", CNeural_activation(nn->layers[layerNum].nodesResults[nodeNum], nn->layers[layerNum].nodes[nodeNum].AF));
                    nn->layers[layerNum].weightedSum[nodeNum] = nn->layers[layerNum].nodesResults[nodeNum];
                    nn->layers[layerNum].nodesResults[nodeNum] =
                        CNeural_activation(nn->layers[layerNum].nodesResults[nodeNum], nn->layers[layerNum].nodes[nodeNum].AF);
                    // printf("\n");
                }

                // printf("\n");
            }

            nn->loss += CNeural_loss(nn->layers[nn->nLayers - 1].nodesResults, labels[label], nn->outShape, nn->lf);

            // TODO implement optimizer flexibility (currently only gradient des.)
            CNeural_derivatives(nn, inputs[label], labels[label], "mse");
            // printf("\n");

            for (int layerNum = 0; layerNum < nn->nLayers; layerNum++) { // clear after each label
                CNeural_clear_nodeResults(nn, layerNum);
            }
        }
        nn->loss = nn->loss / (float) nn->nLabels;
        printf("Loss: %f\n", nn->loss);
        printf("\n");

        if (nn->loss < earlyStopLoss) { // early stopping
            return;
        }
        CNeural_update_weights(nn);


    }
}

/**
 * Activation function. Helper function to CNeural_train.
 * 
 * @param input value to pass through the specified activation
 * @param af a string corresponding to an activation function, values: "none", "sigmoid", "tanh", "relu", default: "relu"
 * @return processed value by the activation
*/
float CNeural_activation(float input, string af) {
    if (strcmp(af, "none") == 0) return input;

    if (strcmp(af, "sigmoid") == 0) {
        return 1 / (1 + expf(-input));
    }
    if (strcmp(af, "tanh") == 0) {
        return tanhf(input);
    }
    if (strcmp(af, "relu") == 0) {
        return fmaxf(0, input);
    }
    printf("Warning: Unknown activation function. Training results might not be optimal!\n");
    printf("Defaulting to ReLu\n");
    return fmaxf(0, input);
}

/**
 * Loss function. Helper function to CNeural_train.
 *
 * @param predicted array of predicted values (nodeResult values of the last layer) for 1 training example
 * @param actual array of label values to compare to for 1 training example
 * @param outputShape number of output nodes
 * @param lfn a string corresponding to a loss function, values: "mse", "mae", default: "mse"
 * @return loss value
*/
float CNeural_loss(float predicted[], float actual[], int outputShape, string lfn) {
    if (strcmp(lfn, "mse") == 0) { // Mean Squared Error
        float sum = 0;
        for (int i = 0; i < outputShape; i++) {
            sum += powf(predicted[i] - actual[i], 2);
        }
        return sum;
    }
    if (strcmp(lfn, "mae") == 0) { // Mean Average Error
        float sum = 0;
        for (int i = 0; i < outputShape; i++) {
            sum += fabsf(predicted[i] - actual[i]);
        }
        return sum;
    }

    printf("Warning: Unknown loss function. Training results might not be optimal!\n");
    printf("Defaulting to MSE\n");
    float sum = 0;
    for (int i = 0; i < outputShape; i++) {
        sum += powf(predicted[i] - actual[i], 2);
    }
    return sum;
}

/**
 * Frees all allocated memory of a neural network.
 *
 * @param nn neural network type
*/
void CNeural_free(NeuralNetwork *nn) {
    for (int i = 0; i < nn->nLayers; i++) {
        for (int j = 0; j < nn->layers[i].nNodes; j++) {
            free(nn->layers[i].nodes->weights);
            free(nn->layers[i].nodes->weightDerivatives);
        }
        free(nn->layers[i].nodes);
        free(nn->layers[i].weightedSum);
        free(nn->layers[i].nodesResults);
        free(nn->layers[i].nodesResultsDerivatives);
    }
    free(nn->layers);
}