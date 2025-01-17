/**
 * \file CNeural.h
 * \brief Header file for CNeural, containing data structures and function declarations.
 *
 * \author Dai Duong Le
 * \version: 0.0.1
*/

#ifndef CNEURAL_H
#define CNEURAL_H
typedef char* string;

/** \struct Node
 * \brief Node type
 *
 * Node type in each layer.
*/
typedef struct {
    float *weights; /**< array of weights */
    float bias; /**< bias value */
    float *weightDerivatives; /**< corresponding nudges to weights in gradient descent */
    float biasDerivative; /**< corresponding nudge to the bias in gradient descent */

    string AF; /**< node activation function */
} Node;

/** \struct Layer
 * \brief Layer type.
 *
 * Layer type in a neural network.
*/
typedef struct {
    int nNodes; /**< number of nodes */
    Node *nodes; /**< array of nodes */

    string layerAF; /**< corresponding activations apply to the whole layer, adjust accordingly by changing individual node activations */
    float *weightedSum; /**< weighted sums array of linear combinations before passing through the activation function */
    float *nodesResults; /**< Layer output, weighted sums array that have passed through the activation function as input to the next layer. */
} Layer;

/** \struct NeuralNetwork
 * \brief Neural Network type.
*/
typedef struct {
    int inShape; /**< input shape, the input number of nodes */
    int nLayers; /**< number of layers */
    int outShape; /**< output shape, the final output number of nodes */
    int nLabels; /**< number of labels */
    string lf; /**< loss function */
    float loss; /**< the error rate of the network (after each epoch) */
    string opt; /**< optimizer */
    float lr; /**< learning rate */
    int epochs; /**< number of epochs (forward passes through the whole dataset) */

    Layer *layers; /**< array of layers */
} NeuralNetwork;


int CNeural_init(NeuralNetwork *nn, int inputShape, int outputShape, int numLayers, int layerNumNodes[], string layersAF[], string initMethod);
void CNeural_clear_nodeResults(NeuralNetwork *nn, int layerNum);
int CNeural_wb_init(NeuralNetwork *nn, int layerNum, string option);
void CNeural_train(NeuralNetwork *nn, int numLabels, float inputs[numLabels][nn->inShape], float labels[numLabels][nn->outShape], string lossFunction, string optimizer, float learningRate, int epochs);
float CNeural_activation(float input, string af);
float CNeural_loss(float predicted[], float actual[], int outputShape, string lfn);
void CNeural_free(NeuralNetwork *nn);

void CNeural_derivatives(NeuralNetwork *nn, float inputs[], float labels[], string lossFunction);
float CNeural_af_derivative(float input, string af);
float CNeural_loss_derivative(float predicted, float actual, string lfn);
#endif //CNEURAL_H
