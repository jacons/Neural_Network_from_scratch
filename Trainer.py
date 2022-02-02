import math

from NeuralN import *


class Trainer:

    def __init__(self, tr, vl, l_rate, metric):
        """
        Configure the neural network by hyperparameter and dataset
        :param tr: Training set using for train the network
        :param vl: Validation set using to measure of generalization
        :param l_rate: function of manage weights decay
        :param metric: function to evaluate the error
        """

        self.__tr = tr  # Training set
        self.__vl = vl  # Validation set
        self.__sizeTR: int = len(tr)  # Training set size
        self.__sizeVL: int = len(vl)  # Validation set size

        self.__features = 0
        self.__mB = None  # Array of minibatch

        self.__sequenceMB = None  # Array that contain a sequence of number using by minibatch
        self.__sequenceVL = range(0, self.__sizeVL)  # small improvement for speed up

        self.__batchSize: int = 1  # Bach size for minibatch technique
        self.__maxEpochs = 1000  # Number of epochs

        self.__patience = 0  # number of consecutive epoch for witch validation error increases
        self.__stopping = 0  # countdown for early stopping

        self.__metric = metric  # evaluate error
        self.__l_rate = l_rate  # learning rate function

        self.__mse_tr: list = []  # Train set error's sequence (during the epochs)
        self.__mse_vs: list = []  # Validation set error's sequence (during the epochs)

        self.__nn = None

    def set(self, features: int, lmd: float, alfa: float, batch_size: int,
            max_epochs: int, nesterov: bool, patience: int):
        """
        Initialize hyperparameters for the neural network
        :param features: Number of the input
        :param lmd: lambda value for regularization
        :param alfa:  alfa value for momentum
        :param batch_size: bach size for mini batch technique
        :param max_epochs: max epochs for the training phase
        :param patience: number of consecutive epoch for witch validation error increases
        :param nesterov: boolean value, True -> NAG , False -> CM
        :return: None
        """
        if lmd < 0 or alfa < 0 or max_epochs < 0 or batch_size <= 0 or batch_size > self.__sizeTR:
            print("Bad initialization values")
            exit()

        self.__features = features  # Number of features
        self.__batchSize: int = batch_size  # Bach size for minibatch technique
        self.__maxEpochs: int = max_epochs  # Max number of epochs

        # If patience is zero, we don't take into account the early stopping
        self.__patience: int = self.__maxEpochs if patience == 0 else patience
        self.__stopping = self.__patience

        batches = math.ceil(self.__sizeTR / batch_size)  # Perform the numer of batch to create
        self.__mB = np.array_split(self.__tr, batches)  # splitting dataset
        self.__sequenceMB = np.arange(0, batches, 1)  # create a sequence 1....batches

        self.__nn = NeuralNetwork(features=features, lmd=lmd, alfa=alfa, metric=self.__metric,
                                  nesterov=nesterov)

    def addLayer(self, units, phi, d_phi):
        """
        Adding a layer inside the network, it requires the number of units and the activation function
        for the layer
        :param units: Number of hidden units
        :param phi: Activation function used in feedforward
        :param d_phi:  Derivative of activation function used in backpropagation
        :return: None
        """
        self.__nn.addLayer(neuron=units, phi=phi, d_phi=d_phi)

    def compile(self):
        """
        Initializing networks' weights and other fancy stuff
        :return: None
        """
        self.__nn.kernelInitialization()

    def fit(self):
        """
        Training the network
        :return: None
        """
        print("Start training phase!")

        current = 0  # current epoch
        while current < self.__maxEpochs and self.__stopping != 0:
            # ---------------  Training phase ---------------
            total_error = 0  # Total error respect all training
            np.random.shuffle(self.__sequenceMB)  # make a permutation of sequenceMB

            for mb in self.__sequenceMB:  # for each mini batches
                b_length = len(self.__mB[mb])  # retrieve batch's length

                for k in range(0, b_length):  # for each element in batch "mb"
                    total_error += self.__nn.train(
                        self.__mB[mb][k][0:self.__features], self.__mB[mb][k][self.__features:])

                self.__nn.update(eta=self.__l_rate(current), batch_size=b_length)

            self.__mse_tr.append(total_error / self.__sizeTR)
            # ---------------  Training phase ---------------

            # ---------------  Evaluate Validation ---------------
            total_error = 0  # Total error respect all training
            for i in self.__sequenceVL:
                vl_error = self.__vl[i][self.__features:] - self.__nn.predict(self.__vl[i][0:self.__features])
                total_error += self.__metric(vl_error)
            self.__mse_vs.append(total_error / self.__sizeVL)
            # ---------------  Evaluate Validation ---------------

            # ---------------  Check Early stopping ---------------
            if current > 10:
                if self.__mse_vs[-1] > self.__mse_vs[-2]:
                    self.__stopping -= 1
                elif self.__mse_vs[-1] < 0.0005:
                    self.__stopping = 0
                else:
                    self.__stopping = self.__patience
            # ---------------  Check Early stopping ---------------

            current += 1
        print("Stopped at epoch", current, "of", self.__maxEpochs)
        return

    def measureTestSet(self, testSet):
        """
        After training, we can evaluate the trained network with testSet which is the
        portion of dataset that the network has never seen. (Model assessment)
        :param testSet: examples of testSet
        :return: return the error on testSet
        """
        # ---------------  Evaluate Test ---------------
        total_error = 0  # Total error respect all training
        for ex in testSet: # for each testSet example
            ts_error = ex[self.__features:] - self.__nn.predict(ex[0:self.__features])
            total_error += self.__metric(ts_error)
        # ---------------  Evaluate Test ---------------
        return total_error / len(testSet)  # length of entire test set

    def performBlindTS(self, blindTS):
        """
        Taken a final model, we can perform the result for blind testSet, of course we don't have
        the results by dataset (instead of training,validation and internal test set), indeed we return
        only the predicted value of the trained network
        :param blindTS: examples of blindTestSet
        :return: list of outputs' network
        """
        results = []
        for ex in blindTS:
            results.append(self.__nn.predict(ex))

        return results

    def getSequences(self):
        # return Training and Validation error sequence (During the epochs)
        return self.__mse_tr, self.__mse_vs
