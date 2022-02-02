import numpy as np
from numpy import ndarray


# ---------------------- Learning rate approach ----------------------

# Fixed learning rate
def fixed_lr(eta: float): return lambda x: eta


# Tau linear decay (from deep learning)
def tau_linear_decay(tau: int, etaS: float):
    etaF = etaS * 0.1

    def func(epoch):
        alfa = epoch / tau
        eta_k = (1 - alfa) * etaS + alfa * etaF
        if epoch == 0:  # initial step => initial eta
            return etaS
        elif epoch >= tau or eta_k < etaF:  # last step => fix constant eta
            return etaF
        else:
            return eta_k

    return func


# Classical linear decay
def linear_decay(rate: float, etaS: float):
    etaF = etaS * 0.1

    def func(epoch):
        eta = (1 / (1 + rate * epoch)) * etaS
        if epoch == 0:
            return etaS
        elif eta > etaF:
            return eta
        else:
            return etaF

    return func


# ---------------------- Learning rate approach ----------------------

# ---------------------- Metric ----------------------
# Mean euclidian error
def mee(error: ndarray): return np.linalg.norm(error, 2)


# Mean square error
def mse(error: ndarray): return np.power(np.linalg.norm(error, 2), 2)


# Classification accuracy (monk)
def classification_acc(error: ndarray): return np.where(np.abs(error) <= 1, 1, 0)


# ---------------------- Metric ----------------------

def explodeCombination(dictionary: dict):
    """
    Given a dictionary , this method perform an "Explode combination" , returning
    all possible combination of hyperparameters. For each different value in one specific hyperparameter
    the method adds a configuration.
    :return: None
    """
    mesh = np.array(np.meshgrid(*dictionary.values()))
    return mesh.T.reshape(-1, len(dictionary))


# ---------------------- Chats' style ----------------------
font1 = {'family': 'serif', 'color': 'black', 'size': 8}
font2 = {'family': 'serif', 'color': 'black', 'size': 8}
