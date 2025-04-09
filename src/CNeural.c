/**
 * \file CNeural.c
 * \brief Source file for CNeural, containing function definitions.
 *
 * \author Dai Duong Le
 * \version: 1.0.0
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include "CNeural.h"

#define AFStringSize 10 // TODO malloc for AF strings

/* TODO WeightBias Proper Init, add better init options (Xavier, He etc.)
// TODO Make strings enums?
// TODO Labeling Issue? - classification & Prediction function
// TODO Data Import - MNIST
// TODO Save/Import Weights
// TODO Check input/return value validation
// TODO Error messages only to appear once!
// TODO Check for softmax being used only in the last layer, currently doesn't check if it's used in other layers
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
    nn->loss = 0;

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


float nrandom() {
    float u = (float) rand() / RAND_MAX;
    float v = (float) rand() / RAND_MAX;
    return sqrt(-2.0 * logf(u)) * cosf(2.0 * M_PI * v);
}

double random_uniform(double min, double max) {
    double scale = rand() / (double) RAND_MAX;
    return min + scale * (max - min);
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

    srand(time(NULL)); // init seed random generator

    if (strcmp(option, "zero") == 0) {
        // probably not a good init option
    } else if (strcmp(option, "random") == 0) {
        // for (int i = 0; i < nn->layers[layerNum].nNodes; i++) { // for each NODE in layer init weights
        //     if (layerNum == 0) { // first layer # of weights should = # of inputs
        //         nn->layers[layerNum].nodes[i].weights = malloc(sizeof(float) * (unsigned int) nn->inShape);
        //         nn->layers[layerNum].nodes[i].weightDerivatives = malloc(sizeof(float) * (unsigned int) nn->inShape);
        //         if (nn->layers[layerNum].nodes[i].weights == NULL || nn->layers[layerNum].nodes[i].weightDerivatives == NULL) { printf("Error: Failed to allocate memory!"); return 1; }
        //         for (int j = 0; j < nn->inShape; j++) { // for each WEIGHT in node
        //             nn->layers[layerNum].nodes[i].weights[j] = (float) (-1.0f + 2.0f * rand() / ((double) RAND_MAX + 1.0));
        //             // nn->layers[layerNum].nodes[i].weights[j] = (float) ((-0.3f + 0.6f * rand() / ((double) RAND_MAX + 1.0)) / 10);
        //             nn->layers[layerNum].nodes[i].weightDerivatives[j] = 0;
        //         }
        //     } else {  // # of weights should = previous layer # of nodes
        //         nn->layers[layerNum].nodes[i].weights = malloc(sizeof(float) * (unsigned int) nn->layers[layerNum - 1].nNodes);
        //         nn->layers[layerNum].nodes[i].weightDerivatives = malloc(sizeof(float) * (unsigned int) nn->layers[layerNum - 1].nNodes);
        //         if (nn->layers[layerNum].nodes[i].weights == NULL || nn->layers[layerNum].nodes[i].weightDerivatives == NULL) { printf("Error: Failed to allocate memory!"); return 1; }
        //         for (int j = 0; j < nn->layers[layerNum - 1].nNodes; j++) { // for each WEIGHT in node
        //             nn->layers[layerNum].nodes[i].weights[j] = (float) (-1.0f + 2.0f * rand() / ((double) RAND_MAX + 1.0));
        //             // nn->layers[layerNum].nodes[i].weights[j] = (float) ((-0.3f + 0.6f * rand() / ((double) RAND_MAX + 1.0)) / 10);
        //             nn->layers[layerNum].nodes[i].weightDerivatives[j] = 0;
        //         }
        //     }
        //     nn->layers[layerNum].nodes[i].bias = (float) 0; // bias can be 0
        //     // nn->layers[layerNum].nodes[i].bias = (float) ((-0.3f + 0.6f * rand() / ((double) RAND_MAX + 1.0)) / 1);
        //     nn->layers[layerNum].nodes[i].biasDerivative = 0;
        //     nn->layers[layerNum].nodes[i].AF = nn->layers[layerNum].layerAF; // applies to the whole layer
        // }


        for (int i = 0; i < nn->layers[layerNum].nNodes; i++) { // for each NODE in layer init weights
            if (layerNum == 0) { // first layer # of weights should = # of inputs
                nn->layers[layerNum].nodes[i].weights = malloc(sizeof(float) * (unsigned int) nn->inShape);
                nn->layers[layerNum].nodes[i].weightDerivatives = malloc(sizeof(float) * (unsigned int) nn->inShape);
                if (nn->layers[layerNum].nodes[i].weights == NULL || nn->layers[layerNum].nodes[i].weightDerivatives == NULL) { printf("Error: Failed to allocate memory!"); return 1; }

                double std_dev = sqrt(2.0 / nn->inShape);
                for (int j = 0; j < nn->inShape; j++) { // for each WEIGHT in node
                    // Box-Muller transform for normal distribution
                    double u1 = random_uniform(0.0001, 0.9999);
                    double u2 = random_uniform(0.0001, 0.9999);
                    double z = std_dev * sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);

                    nn->layers[layerNum].nodes[i].weights[j] = (float) z;
                    // nn->layers[layerNum].nodes[i].weights[j] = (float) ((-0.3f + 0.6f * rand() / ((double) RAND_MAX + 1.0)) / 10);
                    nn->layers[layerNum].nodes[i].weightDerivatives[j] = 0;
                }
            } else {  // # of weights should = previous layer # of nodes
                nn->layers[layerNum].nodes[i].weights = malloc(sizeof(float) * (unsigned int) nn->layers[layerNum - 1].nNodes);
                nn->layers[layerNum].nodes[i].weightDerivatives = malloc(sizeof(float) * (unsigned int) nn->layers[layerNum - 1].nNodes);
                if (nn->layers[layerNum].nodes[i].weights == NULL || nn->layers[layerNum].nodes[i].weightDerivatives == NULL) { printf("Error: Failed to allocate memory!"); return 1; }

                double std_dev = sqrt(2.0 / nn->layers[layerNum - 1].nNodes);
                for (int j = 0; j < nn->layers[layerNum - 1].nNodes; j++) { // for each WEIGHT in node
                    // Box-Muller transform for normal distribution
                    double u1 = random_uniform(0.0001, 0.9999);
                    double u2 = random_uniform(0.0001, 0.9999);
                    double z = std_dev * sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
                    nn->layers[layerNum].nodes[i].weights[j] = (float) z;
                    // nn->layers[layerNum].nodes[i].weights[j] = (float) ((-0.3f + 0.6f * rand() / ((double) RAND_MAX + 1.0)) / 10);
                    nn->layers[layerNum].nodes[i].weightDerivatives[j] = 0;
                }
            }
            nn->layers[layerNum].nodes[i].bias = (float) 0; // bias can be 0
            // nn->layers[layerNum].nodes[i].bias = (float) ((-0.3f + 0.6f * rand() / ((double) RAND_MAX + 1.0)) / 1);
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
void CNeural_train(NeuralNetwork *nn, int numLabels, float inputs[numLabels][nn->inShape], float labels[numLabels][nn->outShape], string lossFunction, string optimizer, float learningRate, int epochs, float earlyStopLoss, FILE* lossFile) {
    nn->nLabels = numLabels;
    nn->lf = lossFunction;
    nn->opt = optimizer;
    nn->lr = learningRate;
    nn->epochs = epochs;

    // TODO loss and expand activation functions
    for (int epoch = 1; epoch <= nn->epochs; epoch++) {
        printf("Epoch %d/%d\n", epoch, nn->epochs);
        for (int label = 0; label < numLabels; label++) {
            printf("Label: %d\n", label + 1);

            for (int layerNum = 0; layerNum < nn->nLayers; layerNum++) {
                printf("Layer %d\n", layerNum + 1);

                for (int nodeNum = 0; nodeNum < nn->layers[layerNum].nNodes; nodeNum++) {
                    printf("\tNode %d\n", nodeNum + 1);
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
                    if (layerNum == nn->nLayers - 1) {
                        printf("Bias: %f\n", nn->layers[layerNum].nodes[nodeNum].bias);
                    }

                    nn->layers[layerNum].nodesResults[nodeNum] += nn->layers[layerNum].nodes[nodeNum].bias;
                    // printf("\n");
                    // printf("\t\tAfter bias: %f\n", nn->layers[layerNum].nodesResults[nodeNum]);

                    // printf("\t\tAfter activation: %f\n", CNeural_activation(nn, nn->layers[layerNum].nodesResults[nodeNum], nn->layers[layerNum].nodes[nodeNum].AF, nodeNum));
                    nn->layers[layerNum].weightedSum[nodeNum] = nn->layers[layerNum].nodesResults[nodeNum];
                    nn->layers[layerNum].nodesResults[nodeNum] =
                        CNeural_activation(nn, nn->layers[layerNum].nodesResults[nodeNum], nn->layers[layerNum].nodes[nodeNum].AF, nodeNum);
                    // printf("\n");
                }

                // printf("\n");
            }

            // printf("Before %f\n", nn->loss);
            // printf("loss fn ret: %f\n", CNeural_loss(nn->layers[nn->nLayers - 1].nodesResults, labels[label], nn->outShape, nn->lf));
            int labelTempArr[] = {0, 2, 1};
            if (strcmp(nn->lf, "categorical_cross_entropy") == 0) {
                nn->loss += CNeural_loss(nn->layers[nn->nLayers - 1].nodesResults, labels[label], nn->outShape, nn->lf, labelTempArr[label]);
            } else {
                nn->loss += CNeural_loss(nn->layers[nn->nLayers - 1].nodesResults, labels[label], nn->outShape, nn->lf, (int) labelVal(labels[label], nn->outShape));
            }
            // nn->loss += CNeural_loss(nn->layers[nn->nLayers - 1].nodesResults, labels[label], nn->outShape, nn->lf, (int) labelVal(labels[label], nn->outShape));
            // printf("After %f\n", nn->loss);
            // TODO implement optimizer flexibility (currently only gradient des.)
            CNeural_derivatives(nn, inputs[label], labels[label], nn->lf, (int) labelTempArr[label]);
            // printf("\n");

            for (int layerNum = 0; layerNum < nn->nLayers; layerNum++) { // clear after each label
                CNeural_clear_nodeResults(nn, layerNum);
            }
        }
        if (strcmp(nn->lf, "categorical_cross_entropy") != 0) {
            nn->loss = nn->loss / (float) nn->nLabels;
        }
        if (lossFile != NULL) {
            if (epoch == 1) {
                fprintf(lossFile, "epoch,loss\n");
            }
            fprintf(lossFile, "%d,%.6f\n", epoch, nn->loss);
            fflush(lossFile);
        }
        printf("Loss: %f\n", nn->loss);
        printf("\n");

        if (nn->loss < earlyStopLoss) { // early stopping
            return;
        }
        CNeural_update_weights(nn);

    }
}

void CNeural_train_ptr(NeuralNetwork *nn, int numLabels, char* inputs[], char* labels[], string lossFunction, string optimizer, float learningRate, int epochs, float earlyStopLoss, FILE* lossFile) {
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
                                nn->layers[layerNum].nodes[nodeNum].weights[weightNum] * (float) inputs[label][weightNum]; // adds for each linear combination (weighted sum)
                            // printf("Noderes value: %f\n", nn->layers[0].nodesResults[nodeNum]);
                            // printf("Input value: %d\n",  inputs[label][weightNum]);
                        }
                    } else {  // # of weights should = previous layer # of nodes
                        for (int weightNum = 0; weightNum < nn->layers[layerNum - 1].nNodes; weightNum++) {
                            // if (layerNum == nn->nLayers - 1) {
                            //     printf("\t\tWeight %d: %f \t", weightNum + 1, nn->layers[layerNum].nodes[nodeNum].weights[weightNum]);
                            // }
                            nn->layers[layerNum].nodesResults[nodeNum] +=
                                nn->layers[layerNum].nodes[nodeNum].weights[weightNum] * nn->layers[layerNum - 1].nodesResults[weightNum]; // adds for each linear combination (weighted sum)
                            // printf("Noderes value: %f\n", nn->layers[layerNum].nodesResults[nodeNum]);
                        }
                    }
                    // printf("\t\tBiasder: %f ", nn->layers[0].nodes[nodeNum].biasDerivative);
                    nn->layers[layerNum].nodesResults[nodeNum] += nn->layers[layerNum].nodes[nodeNum].bias;
                    // printf("\n");
                    // printf("\t\tAfter bias: %f\n", nn->layers[layerNum].nodesResults[nodeNum]);

                    // printf("\t\tAfter activation: %f\n", CNeural_activation(nn, nn->layers[layerNum].nodesResults[nodeNum], nn->layers[layerNum].nodes[nodeNum].AF), nodeNum);
                    nn->layers[layerNum].weightedSum[nodeNum] = nn->layers[layerNum].nodesResults[nodeNum];

                    // if (layerNum == nn->nLayers - 1 && label == 40002) {
                    //     printf("%f\n", nn->layers[layerNum].nodesResults[nodeNum]);
                    //     printf(" %f\n", CNeural_activation(nn, nn->layers[layerNum].nodesResults[nodeNum], nn->layers[layerNum].nodes[nodeNum].AF, nodeNum));
                    // }
                    nn->layers[layerNum].nodesResults[nodeNum] =
                        CNeural_activation(nn, nn->layers[layerNum].nodesResults[nodeNum], nn->layers[layerNum].nodes[nodeNum].AF, nodeNum);
                    // if (layerNum == nn->nLayers - 1 && label == 40002) {
                    //     printf("%f\n", nn->layers[layerNum].nodesResults[nodeNum]);
                    //     printf(" %f\n", CNeural_activation(nn, nn->layers[layerNum].nodesResults[nodeNum], nn->layers[layerNum].nodes[nodeNum].AF, nodeNum));
                    //     printf("%f\n", nn->layers[layerNum].nodesResults[nodeNum]);
                    // }
                    // printf("\t\tAfter activation: %f\n", nn->layers[layerNum].nodesResults[nodeNum]);

                    // if (layerNum == nn->nLayers - 1) {
                    //     for (int i = 0; i < nn->layers[layerNum].nNodes; i++) {
                    //         printf("\t\tLoop After activation: %f\n", nn->layers[layerNum].nodesResults[i]);
                    //     }
                    // }
                    // printf("\n");
                }

                // printf("\n");
            }

            nn->loss += CNeural_loss(nn->layers[nn->nLayers - 1].nodesResults, labels[label], nn->outShape, nn->lf, labelVal(labels[label], nn->outShape));

            // TODO implement optimizer flexibility (currently only gradient des.)
            CNeural_derivatives(nn, inputs[label], labels[label], nn->lf, (int) labelVal(labels[label], nn->outShape));
            // printf("\n");

            for (int layerNum = 0; layerNum < nn->nLayers; layerNum++) { // clear after each label
                CNeural_clear_nodeResults(nn, layerNum);
            }
        }

        if (strcmp(nn->lf, "categorical_cross_entropy") != 0) {
            nn->loss = nn->loss / (float) nn->nLabels;
        }
        if (lossFile != NULL) {
            if (epoch == 1) {
                fprintf(lossFile, "epoch,loss\n");
            }
            fprintf(lossFile, "%d,%.6f\n", epoch, nn->loss);
            fflush(lossFile);
        }
        printf("Loss: %f\n", nn->loss);
        printf("\n");

        if (nn->loss < earlyStopLoss) { // early stopping
            return;
        }
        CNeural_update_weights(nn);


    }
}

float labelVal(float label[], int outputShape) {
    int labelVal;
    for (int i = 0; i < outputShape; i++) {
        // printf("%f\n", label[i]);
        if (label[i] != 0) {
            // printf("%d\n", i);
            labelVal = i;
        }
    }
    return labelVal;
}

/**
 * Activation function. Helper function to CNeural_train.
 * 
 * @param input value to pass through the specified activation
 * @param af a string corresponding to an activation function, values: "none", "sigmoid", "tanh", "relu", default: "relu"
 * @return processed value by the activation
*/
float CNeural_activation(NeuralNetwork *nn, float input, string af, int nodeNum) {
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
    if (strcmp(af, "softmax") == 0) {
        // softmax only works when the LAST nodeRes has been calculated, should only be used for the last layer
        if (nn->layers[nn->nLayers - 1].nNodes - 1 == nodeNum) {
            // printf("Entered SOFTMAX");
            //
            // printf("input: %f\n", input);
            // float maxVal = nn->layers[nn->nLayers - 1].nodesResults[0];
            // for (int i = 1; i < nn->layers[nn->nLayers - 1].nNodes; i++) {
            //     if (nn->layers[nn->nLayers - 1].nodesResults[i] > maxVal) {
            //         maxVal = nn->layers[nn->nLayers - 1].nodesResults[i];
            //     }
            // }
            // printf("maxVal: %f\n", maxVal);
            float sum = 0;
            for (int i = 0; i < nn->layers[nn->nLayers - 1].nNodes; i++) {
                // printf("noderes: %f\n", nn->layers[nn->nLayers - 1].nodesResults[i]);
                // printf("eToNoderes: %f\n", expf(nn->layers[nn->nLayers - 1].nodesResults[i]));
                sum += expf(nn->layers[nn->nLayers - 1].nodesResults[i]);
            }
            // printf("sum: %f\n", sum);
            for (int i = 0; i < nn->layers[nn->nLayers - 1].nNodes; i++) {
                // printf("%f\n", expf(nn->layers[nn->nLayers - 1].nodesResults[i]) / sum);
                // printf("inside before %f\n", nn->layers[nn->nLayers - 1].nodesResults[i]);
                nn->layers[nn->nLayers - 1].nodesResults[i] = expf(nn->layers[nn->nLayers - 1].nodesResults[i]) / sum;
                // printf("inside %f\n", nn->layers[nn->nLayers - 1].nodesResults[i]);
                if (i == nn->layers[nn->nLayers - 1].nNodes - 1) {
                    // printf("inside last %f\n", nn->layers[nn->nLayers - 1].nodesResults[i]);
                    return nn->layers[nn->nLayers - 1].nodesResults[i];
                }
            }

            // printf("%f\n", expf(input)/sum);
            // return expf(input)/sum;
        }
    }
    if (strcmp(af, "softmax") != 0) {
        printf("Warning: Unknown activation function. Training results might not be optimal!\n");
        printf("Defaulting to ReLu\n");
        return fmaxf(0, input);
    }
    return input;
}

/**
 * Loss function. Helper function to CNeural_train.
 *
 * @param predicted array of predicted values (nodeResult values of the last layer) for 1 training example
 * @param actual array of label values to compare to for 1 training example
 * @param outputShape number of output nodes
 * @param lfn a string corresponding to a loss function, values: "mse", "mae", "categorical_cross_entropy" default: "mse"
 * @return loss value
*/
float CNeural_loss(float predicted[], float actual[], int outputShape, string lfn, int labelVal) {
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
    if (strcmp(lfn, "categorical_cross_entropy") == 0) {
        // float sum = 0;
        // for (int i = 0; i < outputShape; i++) {
            // printf("predicted: %f\n", predicted[labelVal]);
            // sum = -logf(predicted[labelVal]);
        // }
        // printf("sum: %f\n", sum);
        return -logf(predicted[labelVal]);
    }

    printf("Warning: Unknown loss function. Training results might not be optimal!\n");
    printf("Defaulting to MSE\n");
    float sum = 0;
    for (int i = 0; i < outputShape; i++) {
        sum += powf(predicted[i] - actual[i], 2);
    }
    return sum;
}

void CNeural_predict(NeuralNetwork *nn, float input[]) {
    // Draws Input
    // for (int i = 0; i < 28; i++) {
    //     for (int j = 0; j < 27; j++) {
    //         if (input[i * 28 + j] == 0) {
    //             printf("%d", input[i * 28 + j]);
    //         } else if (input[i * 28 + j] == 1) {
    //             printf("%d", input[i * 28 + j]);
    //         }
    //     }
    //     printf("\n");
    // }
    for (int layerNum = 0; layerNum < nn->nLayers; layerNum++) {
        for (int nodeNum = 0; nodeNum < nn->layers[layerNum].nNodes; nodeNum++) {
            if (layerNum == 0) { // 1st layer # of weights should = # of inputs
                for (int weightNum = 0; weightNum < nn->inShape; weightNum++) {
                    printf("Mul: %f\n", nn->layers[layerNum].nodes[nodeNum].weights[weightNum] * input[weightNum]);
                    nn->layers[layerNum].nodesResults[nodeNum] +=
                        nn->layers[layerNum].nodes[nodeNum].weights[weightNum] * input[weightNum];
                }
            } else {  // # of weights should = previous layer # of nodes
                for (int weightNum = 0; weightNum < nn->layers[layerNum - 1].nNodes; weightNum++) {
                    nn->layers[layerNum].nodesResults[nodeNum] +=
                        nn->layers[layerNum].nodes[nodeNum].weights[weightNum] * nn->layers[layerNum - 1].nodesResults[weightNum]; // adds for each linear combination (weighted sum)
                }
            }
            if (layerNum == nn->nLayers - 1) {
                // printf("Bias: %f\n", nn->layers[layerNum].nodes[nodeNum].bias);
            }
            nn->layers[layerNum].nodesResults[nodeNum] += nn->layers[layerNum].nodes[nodeNum].bias;
            nn->layers[layerNum].nodesResults[nodeNum] = CNeural_activation(nn, nn->layers[layerNum].nodesResults[nodeNum], nn->layers[layerNum].nodes[nodeNum].AF, nodeNum);
        }
    }
    for (int i = 0; i < nn->outShape; i++) {
        printf("predicted: %f\n", nn->layers[nn->nLayers - 1].nodesResults[i]);
    }
    for (int layerNum = 0; layerNum < nn->nLayers; layerNum++) { // clear after each label
        CNeural_clear_nodeResults(nn, layerNum);
    }
    for (int i = 0; i < nn->outShape; i++) {
        printf("predicted: %f\n", nn->layers[nn->nLayers - 1].nodesResults[i]);
    }
}

void CNeural_predict_ptr(NeuralNetwork *nn, char* input) {
    // Draws Input
    // for (int i = 0; i < 28; i++) {
    //     for (int j = 0; j < 27; j++) {
    //         if (input[i * 28 + j] == 0) {
    //             printf("%d", input[i * 28 + j]);
    //         } else if (input[i * 28 + j] == 1) {
    //             printf("%d", input[i * 28 + j]);
    //         }
    //     }
    //     printf("\n");
    // }
    for (int layerNum = 0; layerNum < nn->nLayers; layerNum++) {
        for (int nodeNum = 0; nodeNum < nn->layers[layerNum].nNodes; nodeNum++) {
            if (layerNum == 0) { // 1st layer # of weights should = # of inputs
                for (int weightNum = 0; weightNum < nn->inShape; weightNum++) {
                    // printf("Mul: %f\n", input[weightNum]);
                    nn->layers[layerNum].nodesResults[nodeNum] +=
                        nn->layers[layerNum].nodes[nodeNum].weights[weightNum] * input[weightNum];
                }
            } else {  // # of weights should = previous layer # of nodes
                for (int weightNum = 0; weightNum < nn->layers[layerNum - 1].nNodes; weightNum++) {
                    nn->layers[layerNum].nodesResults[nodeNum] +=
                        nn->layers[layerNum].nodes[nodeNum].weights[weightNum] * nn->layers[layerNum - 1].nodesResults[weightNum]; // adds for each linear combination (weighted sum)
                }
            }
            nn->layers[layerNum].nodesResults[nodeNum] += nn->layers[layerNum].nodes[nodeNum].bias;
            nn->layers[layerNum].nodesResults[nodeNum] = CNeural_activation(nn, nn->layers[layerNum].nodesResults[nodeNum], nn->layers[layerNum].nodes[nodeNum].AF, nodeNum);
        }
    }
    for (int i = 0; i < nn->outShape; i++) {
        printf("predicted: %f\n", nn->layers[nn->nLayers - 1].nodesResults[i]);
    }
    for (int layerNum = 0; layerNum < nn->nLayers; layerNum++) { // clear after each label
        CNeural_clear_nodeResults(nn, layerNum);
    }
}

/**
 * Frees all allocated memory of a neural network.
 *
 * @param nn neural network type
*/
void CNeural_free(NeuralNetwork *nn) {
    for (int i = 0; i < nn->nLayers; i++) {
        // for (int j = 0; j < nn->layers[i].nNodes; j++) {
        //     free(nn->layers[i].nodes->weights);
        //     free(nn->layers[i].nodes->weightDerivatives);
        // }
        free(nn->layers[i].nodes);
        free(nn->layers[i].weightedSum);
        free(nn->layers[i].nodesResults);
        free(nn->layers[i].nodesResultsDerivatives);
    }
    free(nn->layers);
}
