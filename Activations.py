import math
import numpy as np


class Activation:
    def __init__(self, name: str, phi, dPhi):
        """
        Utility class used for manage name,function and its derivative for all activation function
        :param name: Name of activation function, shown in the title of chats
        :param phi: pointer to activation function
        :param dPhi: pointer to derivative of activation function
        """
        self.name = name
        self.phi = phi
        self.dPhi = dPhi


def linear(x): return x


def d_linear(x): return 1


def relu(x): return max(0, x)


def LReLU(x): return x if x >= 0 else 0.01 * x


def d_LReLU(x): return 1 if x >= 0 else 0.01


def d_relu(x): return 0 if x < 0 else 1


def sigmoid(x): return 1 / (1 + np.exp(-x))


def d_sigmoid(x):
    s = sigmoid(x)
    return s * (1 - s)


def tanH(x): return np.tanh(x)


def d_tanH(x): return 1 - math.pow(np.tanh(x), 2)


# Dictionary of activation function used in the grid search configuration
actF = {
    "linear": Activation("linear", linear, d_linear),
    "relu": Activation("ReLu", relu, d_relu),
    "sig": Activation("Sigmoid", sigmoid, d_sigmoid),
    "tan": Activation("tanH", tanH, d_tanH),
    "lRelu": Activation("LReLu", LReLU, d_LReLU),

}
