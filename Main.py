from Activations import actF
from Parser import *
from Validator import *

# ----------------------  GRID SEARCH PARAMETERS ----------------------

configuration = {
    "lr": np.array([0.043]),
    "tau": np.array([350]),
    "nesterov": np.array([True]),
    "momentum": np.array([0.65]),
    "lambda": np.array([0.00005]),
    "patience": np.array([7]),
    "epoch": np.array([20]),
    "batchSize": np.array([1]),
    "af_out": np.array([actF["linear"]]),
    "af_ho": np.array([actF["tan"]]),
    "hiddenUnit": np.array([25])
}
# ----------------------  GRID SEARCH PARAMETERS ----------------------


def taskCup(tr, vl, config):
    """
    This method is user for train a neural network give a specific training and validation set, works only for
    Cup regression task, config represents the hyperparameters
    :param tr: Training set
    :param vl: Validation set
    :param config: Combination of hyperparameters
    :return: pointer to trainer instance
    """
    trainer = Trainer(tr=tr, vl=vl, l_rate=tau_linear_decay(config[1], etaS=float(config[0])), metric=mee)
    trainer.set(features=10, lmd=float(config[4]), alfa=float(config[3]), batch_size=int(config[7]),
                max_epochs=int(config[6]), patience=config[5], nesterov=config[2])
    # Oss. config 8 and 9 are both objects that contains information to activation functions
    # with "dPhi" we refer "derivative of activation function"
    trainer.addLayer(int(config[10]), phi=config[9].phi, d_phi=config[9].dPhi)
    trainer.addLayer(2, phi=config[8].phi, d_phi=config[8].dPhi)

    trainer.compile()  # initialize weights
    trainer.fit()  # training phase
    return trainer


def taskMonk(tr, vl, config):
    """
    This method is user for train a neural network give a specific training and validation set, works only for
    Monk classification task, config represents the hyperparameters
    :param tr: Training set
    :param vl: Validation set
    :param config: Combination of hyperparameters
    :return: pointer to trainer instance
    """
    trainer = Trainer(tr=tr, vl=vl, l_rate=fixed_lr(float(config[0])), metric=mse)
    trainer.set(features=17, lmd=float(config[4]), alfa=float(config[3]), batch_size=int(config[7]),
                max_epochs=int(config[6]), patience=config[5], nesterov=config[2])
    # Oss. config 8 and 9 are both objects that contains information to activation functions
    # with "dPhi" we refer "derivative of activation function"
    trainer.addLayer(int(config[10]), phi=config[9].phi, d_phi=config[9].dPhi)
    trainer.addLayer(1, phi=config[8].phi, d_phi=config[8].dPhi)

    trainer.compile()  # initialize weights
    trainer.fit()  # training phase
    return trainer


def Monk():
    # monk_tr = importMonk("datasets/Monk3/monks-3.train")
    # monk_vl = importMonk("datasets/Monk3/monks-3.test")

    # monk_tr = importMonk("datasets/Monk2/monks-2.train")
    # monk_vl = importMonk("datasets/Monk2/monks-2.test")

    monk_tr = importMonk("datasets/Monk2/monks-1.train")
    monk_vl = importMonk("datasets/Monk2/monks-1.test")

    # Starting the validation phase
    Validation(confg=configuration, cpu=20, task=taskMonk).monkValidation(tr=monk_tr, vl=monk_vl)


def Cup():
    cup_tr = importCUP("datasets/Cup/ML-CUP21-TR.csv")
    cup_ts = importCUP("datasets/Cup/ML-CUP21-TS.csv")

    pathBlindTS = "datasets/Cup/ML-CUP21-BLIND-TS.csv"

    # Starting the validation phase
    v = Validation(confg=configuration, cpu=16, task=taskCup)
    v.cupValidation(trainSet=cup_tr, testSet=cup_ts, blind=True, lastTrain=True, blindTS=pathBlindTS)


if __name__ == '__main__':
    Cup()
