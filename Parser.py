import pandas as pd
import numpy as np


def oneHotEncoder(features):
    """
    OneHotEncoder for monks dataset
    :param features: input vector (6 elements)
    :return: encoded input vector (17 elements)
    """
    # num_features is used to memorize how many value a particular feature could take
    # (ex.) if num_feature[1]=3 we know that the second feature could take only 3 value (1 or 2 or 3)
    num_features = np.array([3, 3, 2, 3, 4, 2])  # attribute value information
    ohe = np.zeros(17)  # prepare an array full of 17 zero where we store the example encoded
    j = 0
    for i in range(len(features)):
        sub = features[i] - 1
        ohe[sub + j] = 1
        j += num_features[i]
    return ohe


def importCUP(dataset):
    """
    Import Cup regression task 's dataset
    :param dataset: dataset 's file
    :return:
    """
    column_name = ['id', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'y1', 'y2']
    dt = pd.read_csv(dataset, sep=",", names=column_name)
    dt.set_index('id', inplace=True)
    return dt.to_numpy()


def importBlindTS(dataset):
    """
    Import Cup regression task 's dataset
    :param dataset: dataset 's file
    :return:
    """
    column_name = ['id', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10']
    dt = pd.read_csv(dataset, sep=",", names=column_name)
    dt.set_index('id', inplace=True)
    return dt.to_numpy()


def importMonk(dataset):
    """
    Import Monk classification task 's dataset
    :param dataset: dataset 's file
    :return:
    """
    column_name = ['y1', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'id']
    features = pd.read_csv(dataset, sep=" ", names=column_name)
    features.set_index('id', inplace=True)
    target = features['y1'].to_numpy()
    target = np.expand_dims(target, 1)

    target = np.vectorize(lambda z: -1 if z == 0 else 1)(target)
    features.drop(columns=['y1'], inplace=True)
    features = features.to_numpy()
    ohe = np.apply_along_axis(oneHotEncoder, 1, features)
    dataset = np.concatenate((ohe, target), axis=1)
    return dataset
