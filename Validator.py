import sys
import time
from datetime import datetime
from pathlib import Path

import Parser
from Factory import *
from matplotlib import pyplot as plt
from multiprocessing import Process, Manager
from Trainer import *


class Validation:

    def __init__(self, confg: dict, task, cpu: int, kFold: int = 4):
        """
        This class handle the validation phase using grid search to seek to the best hyperparameter.
        For each configuration in the grid search we perform the k-fold cross validation in order to return
        well posed empirical error
        :param confg: Dictionary with all possible configuration of hyperparameters
        :param task: Pointer to task function whose performs the effective train
        :param cpu: Number of parallel train
        :param kFold: Number of fold in k-fold cross validation
        """

        self.__k: int = kFold  # numer of kFold, default 4
        self.__task = task  # Task to perform the train
        self.__cpu = cpu  # number of parallel train

        # return all combinations from possible value of hyperparameters,
        # return -> list of array, each array is a combination
        self.__param = explodeCombination(dictionary=confg)
        self.__b_hyperParam = None  # best hyperparameter found

        self.__dFold = None  # k-fold part respect to entire dataset (only for Cup)

        self.__monkTr = None  # Training error (only for monk)
        self.__monkVL = None  # Validation error  (only for monk)
        # (I know it's a bad practice)

        # Directory to save the results
        self.__name = "results/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        Path(self.__name).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def getTitle(param):
        return "eta:" + format(param[0], 'G') + \
               ", tau:" + format(param[1], 'G') + \
               ", nst:" + ("True" if param[2] else "False") + \
               ", alf:" + format(param[3], 'G') + \
               ", lmd:" + format(param[4], 'G') + \
               ", ESP:" + format(param[5], 'G') + \
               ", eph:" + format(param[6], 'G') + \
               ", bts:" + format(param[7], 'G') + \
               ", uts:" + format(param[10], 'G') + \
               ", af1:" + str(param[8].name) + \
               ", af2:" + str(param[9].name)

    def kFoldValidation(self, theta, kFoldResult, gridId):
        """
        This procedure perform the k-fold cross validation given a configuration of hyperparameter theta,
        the k-fold has a deal with dataset split in k's part(by constructor), this method is paralelizabile.
        After k-fold task, it puts inside the dictionary a "Mean of risk (K training)"
        :param theta: One configuration of hyperparameter
        :param kFoldResult: Dictionary to put results
        :param gridId: Index of grid search
        :return: None
        """

        sum_fold_risk = 0  # sum of the risk of all fold , counter
        indexList = np.arange(0, self.__k, 1)  # list of [0,1...k-1], using in order to logic split
        tr_err = np.empty(self.__k, dtype=list)
        vl_err = np.empty(self.__k, dtype=list)

        startTime = time.time()
        for i in range(self.__k):  # for each fold (0...k)

            # we remove i-th index in the list, thus all indexes in the list represent a fold that we have to use
            # in the training set, obliviously the i-th part is the validation
            ids = np.delete(indexList, np.where(indexList == i))

            # now we concatenate the training folds
            tr = np.concatenate((self.__dFold[ids[0]], self.__dFold[ids[1]], self.__dFold[ids[2]]), axis=0)
            # train model with particular hyperparameter values
            trainer = self.__task(tr, self.__dFold[i], theta)
            tr_err[i], vl_err[i] = trainer.getSequences()
            sum_fold_risk += vl_err[i][-1]  # take the risk of validation, just last value in the validation sequence

        # we want to take a mean among different k-fold, it's not easy because may occur early stopping, in this case
        # we should pick just the values witch is different from zeros

        # transform an array of list in an array of len
        lengths = np.vectorize(lambda x: len(x))(tr_err)
        # pick the longest sequence
        maxLength = range(0, max(lengths))
        # create a final array where it will be contained a mean among all different sequences
        means_tr = np.zeros(maxLength[-1] + 1)
        means_vl = np.zeros(maxLength[-1] + 1)

        for i in maxLength:
            count = 0
            # pick value from all lists (if exist and different from zero)
            for idx in indexList:
                if i < lengths[idx] and (tr_err[idx][i] != 0 or vl_err[idx][i] != 0):
                    means_tr[i] += tr_err[idx][i]
                    means_vl[i] += vl_err[idx][i]
                    count += 1
            # count represent the number of "value" different from zero, for each position
            # in the sequence
            means_vl[i] /= count
            means_tr[i] /= count
            i += 1

        endTime = time.time()

        plt.title(self.getTitle(theta), fontdict=font2)
        plt.xlabel("Epochs", fontdict=font1)
        plt.ylabel("M.E.E", fontdict=font1)
        plt.ylim(0, 5)
        plt.grid(color='green', linestyle='--', linewidth=0.4)
        plt.plot(maxLength, means_tr, 'darkmagenta', linewidth='0.8')
        plt.plot(maxLength, means_vl, 'b', linewidth='0.8', linestyle='--')
        plt.legend(["Training set", "Validation set"])
        plt.savefig(self.__name + "/confg" + str(gridId), dpi=1000)

        # after all k-fold, we put in the dictionary the means of the risk
        kFoldResult[gridId] = [means_tr[-1], means_vl[-1], endTime - startTime]
        print("Process", gridId, "Done!")

    def cupValidation(self, trainSet, testSet, blind: bool, lastTrain: bool, blindTS):
        """
        The model selection phase manage the grid search for the cup regression task,
        starting a k-fold "process" for each configuration.
        Oss. the main process (this) and all "k-fold" process shared only a dictionary used for store the empirical
        risk
        :param: dataset entire dataset (feature and target are not split)
        :return: None
        """
        # now perform a splitting in k "fold", only once, merely we split the dataset in k part.
        self.__dFold = np.array_split(trainSet, self.__k)

        kFoldsErrors = Manager().dict()  # define a synchronized and shared variable.

        # ------------------ GRID SEARCH PHASE ------------------
        # for each configuration theta in "param" we perform k-fold
        chunks = np.array_split(self.__param, math.ceil(len(self.__param) / self.__cpu))

        gridId = 0  # grid search id
        for chunk in chunks:
            # Define an array of workers that perform independently the k-fold phase
            workers = np.empty(len(chunk), dtype=Process)
            iteration = 0
            # ------------------ GRID SEARCH PHASE ------------------
            # for each configuration theta in "param" we perform k-fold
            for theta in chunk:
                print("The ", gridId, "th grid search is starting!!!")
                workers[iteration] = Process(target=self.kFoldValidation, args=(theta, kFoldsErrors, gridId))
                workers[iteration].start()
                iteration += 1
                gridId += 1
            # after that we've started all k-fold, we are going to wait the termination
            for i in range(len(chunk)):
                workers[i].join()
            # ------------------ GRID SEARCH PHASE ------------------

        print("Dictionary results\n", kFoldsErrors)
        min_index = 0
        minimum = sys.maxsize  # minimum is max integer

        f = open(self.__name + "/results.txt", "a")
        f.write("Grid Id, [Training error, Validation error, Time elapsed], Hyperparameters\n")
        for key, value in kFoldsErrors.items():

            f.write(
                "id: " + str(key) + " values: " + str(value) +
                " eta: " + str(self.__param[key][0]) + " tau: " + str(self.__param[key][1]) +
                " Nst: " + str(self.__param[key][2]) + " alfa: " + str(self.__param[key][3]) +
                " lmb: " + str(self.__param[key][4]) + " units: " + str(self.__param[key][10]) +
                " epochs: " + str(self.__param[key][6]) + " BS: " + str(self.__param[key][7]) +
                " A.f out: '" + str(self.__param[key][8].name) + "' A.f hidden: '" + str(self.__param[key][9].name) +
                "'\n")

            if value[1] < minimum:
                min_index = key
                minimum = value[1]

        f.close()
        # take index in array of combinations(grid search) where the risk(in the k-fold) in minimum
        self.__b_hyperParam = self.__param[min_index]  # theta start

        if lastTrain:
            # If checked we perform the last train with holdout technique in order to return
            # the best model, if lastTrain is enabled , we can perform also the blindTest
            print("Starting the last training")
            startTime = time.time()
            # Retrain a model with the best hyperparameter using hold out validation
            # because we need a validation part to perform early stopping
            dFold = np.array_split(trainSet, 3)
            tr = np.concatenate((dFold[0], dFold[1]), axis=0)

            trainer = self.__task(tr, dFold[2], self.__b_hyperParam)
            tr_err, vl_err = trainer.getSequences()
            ts_err = trainer.measureTestSet(testSet=testSet)
            endTime = time.time()

            f = open(self.__name + "/final_model.txt", "a")
            f.write("Training error: " + str(tr_err[-1]) + "\n")
            f.write("Validation error: " + str(vl_err[-1]) + "\n")
            f.write("Test set error: " + str(ts_err) + "\n")
            f.write("Time elapsed: " + str(endTime - startTime) + "\n")
            f.write("Hyperparameters eta: " + str(self.__b_hyperParam[0]) + " tau: " + str(self.__b_hyperParam[1]) +
                    " Nst: " + str(self.__b_hyperParam[2]) + " alfa: " + str(self.__b_hyperParam[3]) +
                    " lmb: " + str(self.__b_hyperParam[4]) + " units: " + str(self.__b_hyperParam[10]) +
                    " epochs: " + str(self.__b_hyperParam[6]) + " BS: " + str(self.__b_hyperParam[7]) +
                    " A.f out: '" + str(self.__b_hyperParam[8].name) + "' A.f hidden: '" +
                    str(self.__b_hyperParam[9].name) + "'\n")
            f.close()

            if blind:
                # if checked we perform also the blind test, of course after taken the best
                # trained model
                f = open(self.__name + "/blindTSResults.txt", "a")
                blind = trainer.performBlindTS(blindTS=Parser.importBlindTS(blindTS))
                for e in blind:
                    f.write(str(e[0]) + "," + str(e[1]) + "\n")
                f.close()

            plt.title(self.getTitle(self.__param[min_index]), fontdict=font2)
            plt.xlabel("Epochs", fontdict=font1)
            plt.ylabel("M.E.E", fontdict=font1)
            plt.ylim(0, 5)
            plt.grid(color='green', linestyle='--', linewidth=0.4)
            plt.plot(range(0, len(tr_err)), tr_err, 'darkmagenta', linewidth='0.8')
            plt.plot(range(0, len(vl_err)), vl_err, 'b', linewidth='0.8', linestyle='--')
            plt.legend(["Training set", "Validation set"])
            plt.savefig(self.__name + "/best_model", dpi=1000)
            plt.show()
        return

    def holdOut(self, theta, kFoldResult, gridId):
        """
        HoldOut technique for monk classification task

        :param theta: One configuration of hyperparameter
        :param kFoldResult: Dictionary to put results
        :param gridId: Index of grid search
        :return:
        """
        trainer = self.__task(self.__monkTr, self.__monkVL, theta)
        tr_err, vl_err = trainer.getSequences()

        # take last value of validation test to compare various performance
        kFoldResult[gridId] = vl_err[-1]

        plt.title(self.getTitle(theta), fontdict=font2)
        plt.xlabel("Epochs", fontdict=font1)
        plt.ylabel("M.S.E", fontdict=font1)
        plt.grid(color='green', linestyle='--', linewidth=0.4)
        plt.plot(range(0, len(tr_err)), tr_err, 'darkmagenta', linewidth='0.8')
        plt.plot(range(0, len(vl_err)), vl_err, 'b', linewidth='0.8', linestyle='--')
        plt.legend(["Training set", "Validation set"])
        plt.savefig(self.__name + "/confg" + str(gridId), dpi=1000)

    def monkValidation(self, tr, vl):
        """
        The model selection phase manage the grid search for monk classification task,
        starting a holdOut "process" for each configuration.
        Oss. the main process (this) and all "holdOut" process shared only a dictionary used for store the empirical
        risk
        :param tr training set for monk
        :param vl validation set for monk
        :return: None
        """

        startTime = time.time()

        self.__monkTr = tr
        self.__monkVL = vl

        gridErrors = Manager().dict()  # define a synchronized and shared variable.

        # ------------------ GRID SEARCH PHASE ------------------
        # for each configuration theta in "param" we perform k-fold
        chunks = np.array_split(self.__param, math.ceil(len(self.__param) / self.__cpu))

        gridId = 0
        for chunk in chunks:
            # Define an array of workers that perform independently the k-fold phase
            workers = np.empty(len(chunk), dtype=Process)
            iteration = 0
            # ------------------ GRID SEARCH PHASE ------------------
            # for each configuration theta in "param" we perform k-fold
            for theta in chunk:
                print("The ", gridId, "th grid search is starting!!!")
                workers[iteration] = Process(target=self.holdOut, args=(theta, gridErrors, gridId))
                workers[iteration].start()
                iteration += 1
                gridId += 1
            # after that we've started all k-fold, we are going to wait the termination
            for i in range(len(chunk)):
                workers[i].join()
            # ------------------ GRID SEARCH PHASE ------------------

        print("Results\n", gridErrors)

        minimum = sys.maxsize  # minimum is max integer
        min_index = 0

        # I iterate each value in the dictionary (one element = one grid search's value)
        for key, value in gridErrors.items():
            if value < minimum:  # if the "result" it's better than previous
                min_index = key  # keep the index
                minimum = value  # keep the minimum for compare

        # I use the min_index in order to determinate the best hyperparameters
        self.__b_hyperParam = self.__param[min_index]

        print("best index", min_index)

        # ------------------ HOLD OUT WITH THETA STAR ------------------
        # Retrain a model with the best hyperparameter using hold out validation
        # because we need a validation part to perform early stopping
        trainer = self.__task(tr, vl, self.__b_hyperParam)
        tr_err, vl_err = trainer.getSequences()

        plt.title(self.getTitle(self.__param[min_index]), fontdict=font2)
        plt.plot(range(0, len(tr_err)), tr_err, 'darkmagenta', linewidth='0.8')
        plt.plot(range(0, len(vl_err)), vl_err, 'b', linewidth='0.8', linestyle='--')
        plt.legend(["Training set", "Validation set"])
        plt.xlabel("Epochs", fontdict=font1)
        plt.ylabel("M.S.E", fontdict=font1)
        plt.grid(color='green', linestyle='--', linewidth=0.4)
        plt.savefig(self.__name + "/best_model", dpi=1000)
        plt.show()
        print("BEST RESULT ACHIEVED", vl_err[-1])
        endTime = time.time()

        f = open(self.__name + "/results.txt", "a")

        results = "training value result: " + str(tr_err[-1]) + "\n" + \
                  "test value result: " + str(vl_err[-1]) + "\n" + \
                  "time elapsed: " + str(endTime - startTime)
        f.write(results)
        f.close()

        # ------------------ HOLD OUT WITH THETA STAR ------------------

        return
