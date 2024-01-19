# Scratch-Neural-Net
This is a simple implementation of a neural network from scratch in C. The neural network consists of an input layer with 2 neurons, a hidden layer with 2 neurons, and an output layer with 1 neuron. The network is trained to learn a basic XNOR logic gate ( NOT(p XOR q) ).
<img width="68" alt="Screen Shot 2024-01-19 at 1 21 10 PM" src="https://github.com/JoeShans21/Scratch-Neural-Net/assets/31589578/623dcf3c-c97f-4953-8eae-e555dedb28ce">


<h1>Overview</h1>
The neural network is trained using a basic backpropagation algorithm with a sigmoid activation function. Here's an overview of the code's structure and functionality:

* The sigmoid function computes the sigmoid activation of a given input.
* The dSigmoid function computes the derivative of the sigmoid activation function.
* The init_weights function initializes the weights with random values between 0 and 1.
* The shuffle function shuffles an array randomly.
* Macros are used to define the network architecture, including the number of input, hidden, and output neurons, as well as the number of training sets.
* Training data (trainingInputs) and target outputs (trainingOutputs) are defined.
* Weight matrices and bias arrays are initialized with random values.
* The network is trained using a specified learning rate and a defined number of epochs.
* During training, the forward pass is computed to make predictions, and then the backpropagation algorithm is used to adjust weights and biases.
* After training, the final weights and biases of the hidden and output layers are printed to the console.
