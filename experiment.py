import time, sys
import numpy as np
import argparse

import torch

from src.classifier import Classifier
from src.tester import set_reproducible, train_and_eval

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-n', '--n_runs', help='Number of runs.', type=int, default=5)
    argparser.add_argument('-g', '--gpu', help='GPU device id on which to run the model', type=int)
    args = argparser.parse_args()
    device_name = "cpu" if args.gpu is None else f"cuda:{args.gpu}"
    device = torch.device(device_name)
    n_runs = args.n_runs
    set_reproducible()
    datadir = "data//"
    trainfile =  datadir + "traindata.csv"
    devfile =  datadir + "devdata.csv"
    testfile = None
    # testfile = datadir + "testdata.csv"

    # Runs
    start_time = time.perf_counter()
    devaccs = []
    testaccs = []
    for i in range(1, n_runs+1):
        classifier =  Classifier()
        devacc, testacc = train_and_eval(classifier, trainfile, devfile, testfile, i, device)
        devaccs.append(np.round(devacc,2))
        testaccs.append(np.round(testacc,2))
    print('\nCompleted %d runs.' % n_runs)
    total_exec_time = (time.perf_counter() - start_time)
    print("Dev accs:", devaccs)
    print("Test accs:", testaccs)
    print()
    print("Mean Dev Acc.: %.2f (%.2f)" % (np.mean(devaccs), np.std(devaccs)))
    print("Mean Test Acc.: %.2f (%.2f)" % (np.mean(testaccs), np.std(testaccs)))
    print("\nExec time: %.2f s. ( %d per run )" % (total_exec_time, total_exec_time / n_runs))