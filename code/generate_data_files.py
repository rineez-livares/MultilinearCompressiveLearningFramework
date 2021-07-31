#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Author: Rineez Ahmed
@Email: rineez@gmail.com
"""
import exp_configurations as Conf
import numpy as np
import os
import sys
import getopt
import pickle

DATA_DIR = Conf.DATA_DIR


def main(argv):
    try:
        opts, args = getopt.getopt(argv, "h", ['ds='])

    except getopt.GetoptError:
        sys.exit(2)

    for opt, arg in opts:
        if opt == '--ds':
            dataset = str(arg)
            x_train, y_train, x_val, y_val, x_test, y_test = generate_for_dataset(dataset)
            print('x_train', str(x_train))
            np.save(os.path.join(Conf.DATA_DIR, dataset + '_x_train.npy'), x_train)
            print('y_train', str(y_train))
            np.save(os.path.join(Conf.DATA_DIR, dataset + '_y_train.npy'), y_train)
            print('x_val', str(x_val))
            np.save(os.path.join(Conf.DATA_DIR, dataset + '_x_val.npy'), x_val)
            print('y_val', str(y_val))
            np.save(os.path.join(Conf.DATA_DIR, dataset + '_y_val.npy'), y_val)
            print('x_test', str(x_test))
            np.save(os.path.join(Conf.DATA_DIR, dataset + '_x_test.npy'), x_test)
            print('y_test', str(y_test))
            np.save(os.path.join(Conf.DATA_DIR, dataset + '_y_test.npy'), y_test)


def generate_for_dataset(dataset):
    print('Trying to load from dataset', dataset)
    if(dataset == 'cifar-10'):
        return get_cifar10_data()


def unpickle(file):
    return pickle.load(file, encoding='latin1')


# =========== CIFAR-10 Loaders Start ============ Reference: https://stackoverflow.com/a/51694265/569439
def load_cifar_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb') as fo:
        datadict = unpickle(fo)
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000,3072)
        Y = np.array(Y)
        return X, Y


def load_cifar10(ROOT):
    """ load all of cifar """
    xs = []
    ys = []
    for b in range(1,6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
        X, Y = load_cifar_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_cifar_batch(os.path.join(ROOT, 'test_batch'))
    return Xtr, Ytr, Xte, Yte


def get_cifar10_data(num_training=49000, num_validation=1000, num_test=10000):
    # Load the raw CIFAR-10 data
    cifar10_dir = os.path.join(DATA_DIR, 'cifar-10')
    X_train, y_train, X_test, y_test = load_cifar10(cifar10_dir)

    # Subsample the data
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    x_train = X_train.astype('float32')
    x_test = X_test.astype('float32')

    x_train /= 255
    x_test /= 255

    return x_train, y_train, X_val, y_val, x_test, y_test
# =========== CIFAR-10 Loaders End ============


if __name__ == "__main__":
    main(sys.argv[1:])
