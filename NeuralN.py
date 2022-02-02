import numpy as np
from numpy import ndarray


class NeuralNetwork:
    def __init__(self, features: int, lmd: float, alfa: float, metric, nesterov: bool):
        """
        This class represents the core of project, it has a deal with typical concepts of neural network
        feedforward, backpropagation ecc... It's necessary that remain as small and easy as possible
        :param features: Number of input, network features
        :param lmd: lambda value for regularization
        :param alfa: alfa value for momentum
        :param metric: function to evaluate the error
        :param nesterov : True for apply momentum technique
        """

        self.__nLayer: int = 1  # Number of layer, by default is one
        self.__layers: list = [features]  # Array of layers

        self.__activation: list = []  # Array of activation function
        self.__d_activation: list = []  # Array of derivative of activated function

        self.__lambda: float = lmd  # Hyperparameter of Tikhonov regularization
        self.__alfa: float = alfa  # Hyperparameter of Momentum (Nesterov and Classical)

        self.__w = None  # Tensor of weights
        self.__w_deltas = None  # Tensor of delta weights (build-up)
        self.__w_v_prev = None  # Tensor of the previous weights used for momentum
        self.__w_v = None

        self.__b = None  # Tensor of bias (build-up)
        self.__b_deltas = None  # Tensor of delta bias
        self.__b_v_prev = None  # Tensor of the previous bias used for momentum
        self.__b_v = None

        self.__net = None  # Matrix of array that represent the output layer
        self.__phi_net = None  # Matrix of array that represent the output layer activated

        self.__metric = metric  # Function to evaluate the error

        self.__nesterov: bool = nesterov  # TRUE -> apply nesterov momentum, FALSE -> Classical momentum

        # using to speed up the executions
        self.__feedSequence = None
        self.__backSequence = None

        #   net[]   -> just the result of multiplication between "weights" and "input"
        #   phi_net[] -> represent "input" for the next layer (hidden layer activated and ready to use)
        #   weights[]  -> matrix for the current layer
        #   bias[]     -> bias

    def addLayer(self, neuron: int, phi, d_phi):
        """
        Adding layer to Neural Network
        :param neuron: Number of perceptron
        :param phi: pointer to activation function
        :param d_phi: pointer to derivative of activation function
        :return: None
        """
        self.__nLayer += 1
        self.__layers.append(neuron)
        self.__activation.append(phi)
        self.__d_activation.append(d_phi)

    def kernelInitialization(self):
        """
        Initializing all matrix's weights
        :return: None
        """
        # Define an array of matrices one for each layer
        self.__w = np.empty(self.__nLayer - 1, dtype=ndarray)
        self.__w_deltas = np.empty(self.__nLayer - 1, dtype=ndarray)
        self.__w_v = np.empty(self.__nLayer - 1, dtype=ndarray)
        self.__w_v_prev = np.empty(self.__nLayer - 1, dtype=ndarray)

        self.__b = np.empty(self.__nLayer - 1, dtype=ndarray)
        self.__b_deltas = np.empty(self.__nLayer - 1, dtype=ndarray)
        self.__b_v = np.empty(self.__nLayer - 1, dtype=ndarray)
        self.__b_v_prev = np.empty(self.__nLayer - 1, dtype=ndarray)

        # Intermediate results of neural network, they are used for backpropagation technique
        self.__net = np.empty(self.__nLayer - 1, dtype=ndarray)
        self.__phi_net = np.empty(self.__nLayer, dtype=ndarray)

        self.__feedSequence = range(0, self.__nLayer - 1)
        self.__backSequence = range(self.__nLayer - 2, -1, -1)

        for layer in self.__feedSequence:
            init = np.array([-0.7, 0.7]) * (2 / self.__layers[layer])
            self.__w[layer] = np.random.uniform(init[0], init[1], (self.__layers[layer + 1], self.__layers[layer]))
            self.__b[layer] = np.random.uniform(-0.1, 0.1, (self.__layers[layer + 1]))

            self.__w_deltas[layer] = np.zeros((self.__layers[layer + 1], self.__layers[layer]))
            self.__w_v_prev[layer] = np.zeros((self.__layers[layer + 1], self.__layers[layer]))

            self.__b_deltas[layer] = np.zeros((self.__layers[layer + 1]))
            self.__b_v_prev[layer] = np.zeros((self.__layers[layer + 1]))

            self.__w_v[layer] = np.zeros((self.__layers[layer + 1], self.__layers[layer]))
            self.__b_v[layer] = np.zeros((self.__layers[layer + 1]))

    def predict(self, hidden: ndarray):
        """
        Given an input, predict a results base on current neural network
        :param hidden: input vector
        :return: output vector
        """
        for layer in self.__feedSequence:
            hidden = np.dot(self.__w[layer], hidden) + self.__b[layer]
            hidden = np.vectorize(self.__activation[layer])(hidden)
        return hidden

    def train(self, feature: ndarray, target: ndarray):
        """
        Given an input and target vectors, perform the backpropagation to improve the network
        :param feature: input vector
        :param target:  target vector
        :return: error between feature and target
        """
        # we assume that "input" is a result from another (magic) layer

        self.__phi_net[0] = feature

        # ------------------------ Feedforward technique ------------------------
        for layer in self.__feedSequence:
            self.__net[layer] = np.dot(self.__w[layer] - self.__w_v[layer], self.__phi_net[layer]) \
                                + self.__b[layer] - self.__b_v[layer]
            # why is "l+1"? because we suppose that phi_net[0] is equal to input
            self.__phi_net[layer + 1] = np.vectorize(self.__activation[layer])(self.__net[layer])
        # ------------------------ Feedforward technique ------------------------

        # ------------------------ Output Error ------------------------
        # Error: target - result by feedforward, the result is the last hidden_activated (see constructor)
        error_output = target - self.__phi_net[-1]
        error = self.__metric(error_output)
        # ------------------------ Output Error ------------------------

        # ------------------------ Backpropagation ------------------------
        for layer in self.__backSequence:
            phi_derivative = np.vectorize(self.__d_activation[layer])(self.__net[layer])
            # Perform sigma. Oss. error_output for the first iteration is (t-y) , next is W'*sigma(previous)
            sigma = error_output * phi_derivative
            w_deltas = np.outer(sigma, self.__phi_net[layer])
            self.__w_deltas[layer] += w_deltas
            self.__b_deltas[layer] += sigma
            error_output = np.dot(self.__w[layer].T - self.__w_v[layer].T, error_output)
        # ------------------------ Backpropagation ------------------------
        return error

    def update(self, eta: float, batch_size: int):
        """
        Update weights by delta weights accumulated
        :return: None
        """
        for layer in self.__feedSequence:
            # dividing the accumulated weights by numer of elements for each minibatch (normalizing)
            self.__w_deltas[layer] /= batch_size
            self.__b_deltas[layer] /= batch_size

            # apply the classical momentum and subtracting delta weights
            dw = self.__alfa * self.__w_v_prev[layer] + eta * self.__w_deltas[layer]  # param weights
            db = self.__alfa * self.__b_v_prev[layer] + eta * self.__b_deltas[layer]  # param bias

            # finally, update the weights meanwhile applying regularization
            self.__w[layer] += dw - self.__lambda * self.__w[layer]
            self.__b[layer] += db - self.__lambda * self.__b[layer]

            # saving the previous weights used for momentum
            self.__w_v_prev[layer] = dw
            self.__b_v_prev[layer] = db

            if self.__nesterov:
                self.__w_v[layer] = self.__alfa * dw
                self.__b_v[layer] = self.__alfa * db

            # clearing accumulation matrices for the next mini batch
            self.__w_deltas[layer].fill(0)
            self.__b_deltas[layer].fill(0)
