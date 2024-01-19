#include <stdlib.h>
#include <stdio.h>
#include <math.h>

double sigmoid(double x) { return 1 / (1+exp(-x));}
double dSigmoid(double x) { return x * (1-x);}
double init_weights() {return ((double)rand()) / ((double)RAND_MAX); }

void shuffle(int *array, size_t n) {
    if (n > 1) {
        size_t i;
        for (i = 0; i < n; i++) {
            size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
            int t = array[j];
            array[j] = array[i];
            array[i] = t;
        }
    }
}

#define inputs 2
#define hiddenNodes 2
#define outputNodes 1
#define trainingSets 4

int main() {
    const double learningRate = 0.1f;

    double hiddenLayer[hiddenNodes];
    double outputLayer[outputNodes];

    double hiddenLayerBias[hiddenNodes];
    double outputLayerBias[outputNodes];

    double hiddenWeights[inputs][hiddenNodes];
    double outputWeights[outputNodes][hiddenNodes];


    double trainingInputs[trainingSets][inputs] = {{0.0f, 0.0f},{1.0f, 0.0f},{0.0f, 1.0f},{1.0f, 1.0f}};
    double trainingOutputs[trainingSets][outputNodes] = {{1.0f},{0.0f},{0.0f},{1.0f}};



    for(int i = 0; i < inputs; i++) {
        for(int j = 0; j < hiddenNodes; j++) {
            hiddenWeights[i][j] = init_weights();
        }
    }
    for(int i = 0; i < hiddenNodes; i++) {
        for(int j = 0; j < outputNodes; j++) {
            outputWeights[i][j] = init_weights();
        }
    }
    for(int i = 0; i < outputNodes; i++) {
        outputLayerBias[i] = init_weights();
    }

    int trainingSetOrder[] = {0, 1, 2, 3};
    int epochs = 100000;

    //Train the neural network
    for(int epoch = 0; epoch < epochs; epoch++) {
        shuffle(trainingSetOrder, trainingSets);
        for(int x = 0; x < trainingSets; x++) {
            int i = trainingSetOrder[x];
            //Forward pass

            //Compute hidden layer activation
            for(int j = 0; j < hiddenNodes; j++) {
                double activation = hiddenLayerBias[j];
                for(int k = 0; k < inputs; k++) {
                    activation += trainingInputs[i][k] * hiddenWeights[k][j];
                }
                hiddenLayer[j] = sigmoid(activation);
            }

            //Compute output layer activation
            for(int j = 0; j < outputNodes; j++) {
                double activation = outputLayerBias[j];
                for(int k = 0; k < hiddenNodes; k++) {
                    activation += hiddenLayer[k] * outputWeights[k][j];
                }
                outputLayer[j] = sigmoid(activation);
            }

            printf("Input: %g %g Output: %g Predicted Output: %g \n", trainingInputs[i][0], trainingInputs[i][1], outputLayer[0], trainingOutputs[i][0]);

            //Backpropagation

            //Change in output weights

            double deltaOutput[outputNodes];
            for(int j = 0; j < outputNodes; j++) {
                double error = (trainingOutputs[i][j]-outputLayer[j]);
                deltaOutput[j] = error * dSigmoid(outputLayer[j]);
            }

            double deltaHidden[hiddenNodes];
            for(int j = 0; j < hiddenNodes; j++) {
                double error = 0.0f;
                for(int k = 0; k < outputNodes; k++) {
                    error += deltaOutput[k] * outputWeights[j][k];
                }
                deltaHidden[j] = error * dSigmoid(hiddenLayer[j]);
            }

            //Apply change in output weights
            for(int j = 0; j < outputNodes; j++) {
                outputLayerBias[j] += deltaOutput[j] * learningRate;
                for(int k = 0; k < hiddenNodes; k++) {
                    outputWeights[k][j] += hiddenLayer[k] * deltaOutput[j] * learningRate;
                }
            }

            //Apply change in hidden weights
            for(int j = 0; j < hiddenNodes; j++) {
                hiddenLayerBias[j] += deltaHidden[j] * learningRate;
                for(int k = 0; k < inputs; k++) {
                    hiddenWeights[k][j] += trainingInputs[i][k] * deltaHidden[j] * learningRate;
                }
            }


        }
    }
    fputs("\n Final Hidden Weights\n[ ", stdout);
    for(int j = 0; j < hiddenNodes; j++) {
        fputs("[ ", stdout);
        for(int k = 0; k < inputs; k++) {
            printf("%f ", hiddenWeights[k][j]);
        }
        fputs("] ", stdout);
    }
    fputs(" ]\n Final Hidden Biases\n[ ", stdout);
    for(int j = 0; j < hiddenNodes; j++) {
        printf("%f ", hiddenLayerBias[j]);
    }

    fputs(" ]\n Final Output Weights\n[ ", stdout);
    for(int j = 0; j < outputNodes; j++) {
        fputs("[ ", stdout);
        for(int k = 0; k < hiddenNodes; k++) {
            printf("%f ", outputWeights[k][j]);
        }
        fputs(" ]", stdout);
    }

    fputs(" ]\n Final Output Biases\n[ ", stdout);
    for(int j = 0; j < outputNodes; j++) {
        printf("%f", outputLayerBias[j]);
    }

    fputs(" ] \n", stdout);

    return 0;



}


