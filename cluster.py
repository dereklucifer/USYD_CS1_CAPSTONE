from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark.mllib.clustering import KMeans

import datetime
import math
import numpy as np

# Cluster

def clu_compare_with_static_model(training_data, test_data, modelname):
    model = KMeans
    print('*' * 100)
    start = datetime.datetime.now()
    full_model = model.train(training_data, k=5, seed=3)
    print('static model training time:', datetime.datetime.now() - start)

    # retrain model

    print('*' * 100)
    retrain_data, add_data = training_data.randomSplit([0.99, 0.01])
    start = datetime.datetime.now()
    add_model = model.train(retrain_data, k=5, seed=3)
    print('99% data training time', datetime.datetime.now() - start)
    start = datetime.datetime.now()
    add_model = model.train(add_data, k=5, seed=3, initialModel=add_model)
    print('1% data training time', datetime.datetime.now() - start)
    full_center = full_model.centers
    add_center = add_model.centers
    full_center = np.array(full_center).reshape(-1)
    add_center = np.array(add_center).reshape(-1)
    rmse = math.sqrt(sum((x - y) ** 2 for x, y in zip(full_center, add_center)) / len(add_center))
    print('difference between cluster center:', rmse)
    print('*' * 100)

def clu_static_with_different_data_size(training_data, test_data, modelname):
    check_list = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1]
    y_true = test_data.map(lambda row: int(row.label)).collect()
    print('*' * 100)
    model = KMeans

    compare_model = model.train(training_data, k=5, seed=3)
    compare_center = compare_model.centers
    compare_center = np.array(compare_center).reshape(-1)
    for i in check_list:
        training, _ = training_data.randomSplit([i, 1 - i])
        start = datetime.datetime.now()
        KM = model.train(training, k=5, seed=3)
        print('training time:', datetime.datetime.now() - start)
        add_center = KM.centers
        add_center = np.array(add_center).reshape(-1)
        rmse = math.sqrt(sum((x - y) ** 2 for x, y in zip(compare_center, add_center)) / len(add_center))
        print(i, "of raw data: ", rmse)
        print('*' * 100)

def clu_iter_with_different_training_time(training_data, test_data, modelname):
    model = KMeans
    check_list = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1]
    KM_full_model = model.train(training_data, k=5, seed=3)
    full_center = KM_full_model.centers
    full_center = np.array(full_center).reshape(-1)
    for i in check_list:
        training, _ = training_data.randomSplit([i, 1 - i])
        KM = model.train(training, k=5, seed=3)
        add_center = KM.centers
        add_center = np.array(add_center).reshape(-1)
        rmse = math.sqrt(sum((x - y) ** 2 for x, y in zip(full_center, add_center)) / len(add_center))
        print(rmse)

def clu_iter_with_different_training_size(training_data, test_data, modelname):
    check_list = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1]
    model = KMeans
    KM_full_model = model.train(training_data, k=5, seed=3)
    full_center = KM_full_model.centers
    full_center = np.array(full_center).reshape(-1)
    for i in check_list:
        times = int(1 / i)
        for time in range(times):
            training, _ = training_data.randomSplit([i, 1 - i])
            if (time == 0):
                KM = model.train(training, k=5, seed=3)
            else:
                KM = model.train(training, k=5, seed=3, initialModel=KM)
        add_center = KM.centers
        add_center = np.array(add_center).reshape(-1)
        rmse = math.sqrt(sum((x - y) ** 2 for x, y in zip(full_center, add_center)) / len(add_center))
        print(rmse)

def clu_iter_with_different_thread(training_data, test_data, modelname):
    model = KMeans
    print("*" * 100)
    for i in range(5):
        num = pow(2, i)
        print("Core number is:", num)
        sc.stop()
        sc = SparkContext("local[{}]".format(num), "CAPSTONE_{}".format(num)).getOrCreate()
        spark = SparkSession.builder.master("local[{}]".format(num)).appName("CAPSTONE_{}".format(num)).getOrCreate()
        start = datetime.datetime.now()
        full_model = model.train(training_data, k=5, seed=3)
        print("full data: ", datetime.datetime.now() - start)
        training_data, add_data = training_data.randomSplit([0.99, 0.01])
        start = datetime.datetime.now()
        add_model = model.train(training_data, k=5, seed=3)
        print("99% data: ", datetime.datetime.now() - start)
        start = datetime.datetime.now()
        add_model = model.train(add_data, k=5, seed=3, initialModel=add_model)
        print("1% Data: ", datetime.datetime.now() - start)
        print("*" * 100)