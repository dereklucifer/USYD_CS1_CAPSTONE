from pyspark.sql import SparkSession
from pyspark import SparkContext

from pyspark.mllib.regression import LinearRegressionWithSGD, RidgeRegressionWithSGD

import datetime
import math


# Regression

def reg_compare_with_static_model(training_data, test_data, modelname):
    if modelname.__contains__("LR"):
        model = LinearRegressionWithSGD
    elif modelname.__contains__("RR"):
        model = RidgeRegressionWithSGD

    y_true = test_data.map(lambda row: int(row.label)).collect()
    print('*' * 100)
    start = datetime.datetime.now()
    full_model = model.train(training_data, iterations=1000, step=1e-4, miniBatchFraction=1)
    print('static model training time:', datetime.datetime.now() - start)
    y_full_pred = full_model.predict(test_data.map(lambda row: row.features)).map(lambda row: int(row)).collect()
    full_rmse = math.sqrt(sum((x - y) ** 2 for x, y in zip(y_full_pred, y_true)) / len(y_true))
    print('static model training accuracy:', full_rmse)

    # retrain model

    print('*' * 100)
    retrain_data, add_data = training_data.randomSplit([0.99, 0.01])
    start = datetime.datetime.now()
    add_model = model.train(retrain_data, iterations=1000, step=1e-4, miniBatchFraction=1)
    print('99% data training time', datetime.datetime.now() - start)
    start = datetime.datetime.now()
    add_model = model.train(add_data, iterations=1000, step=1e-4, miniBatchFraction=1,
                            initialWeights=add_model.weights)
    print('1% data training time', datetime.datetime.now() - start)
    y_add_pred = add_model.predict(test_data.map(lambda row: row.features)).map(lambda row: int(row)).collect()
    add_rmse = math.sqrt(sum((x - y) ** 2 for x, y in zip(y_add_pred, y_true)) / len(y_true))
    print('retrain model training accuracy:', add_rmse)
    print('*' * 100)


def reg_static_with_different_data_size(training_data, test_data, modelname):
    check_list = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1]
    y_true = test_data.map(lambda row: int(row.label)).collect()
    print('*' * 100)
    if modelname.__contains__("LR"):
        model = LinearRegressionWithSGD
    elif modelname.__contains__("RR"):
        model = RidgeRegressionWithSGD

    for i in check_list:
        training, _ = training_data.randomSplit([i, 1 - i])
        start = datetime.datetime.now()
        full_model = model.train(training, iterations=100)
        print('training time:', datetime.datetime.now() - start)
        y_pred = full_model.predict(test_data.map(lambda row: row.features)).map(lambda row: int(row)).collect()
        iter_rmse = math.sqrt(sum((x - y) ** 2 for x, y in zip(y_pred, y_true)) / len(y_true))
        print(i, "of raw data: ", iter_rmse)
        print('*' * 100)


def reg_iter_with_different_training_time(training_data, test_data, modelname):
    check_list = [1, 2, 4, 8, 16, 32, 64, 100]
    y_true = test_data.map(lambda row: int(row.label)).collect()
    print('*' * 100)
    if modelname.__contains__("LR"):
        model = LinearRegressionWithSGD
    elif modelname.__contains__("RR"):
        model = RidgeRegressionWithSGD

    for i in range(100):
        training, _ = training_data.randomSplit([0.01, 0.99])
        if (i == 0):
            iter_model = model.train(training, iterations=100)
        else:
            iter_model = model.train(training_data, iterations=100, initialWeights=iter_model.weights)
        if (i + 1) in check_list:
            y_pred = iter_model.predict(test_data.map(lambda row: row.features)).map(lambda row: int(row)).collect()
            iter_rmse = math.sqrt(sum((x - y) ** 2 for x, y in zip(y_pred, y_true)) / len(y_true))
            print((i + 1), "% of raw data: ", iter_rmse)


def reg_iter_with_different_training_size(training_data, test_data, modelname):
    check_list = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1]
    y_true = test_data.map(lambda row: int(row.label)).collect()
    print('*' * 100)
    if modelname.__contains__("LR"):
        model = LinearRegressionWithSGD
    elif modelname.__contains__("RR"):
        model = RidgeRegressionWithSGD
    for i in check_list:
        times = int(1 / i)
        for time in range(times):
            training, _ = training_data.randomSplit([i, 1 - i])
            if (time == 0):
                iter_model = model.train(training, iterations=1000, step=1e-4, miniBatchFraction=1)
            else:
                iter_model = model.train(training, initialWeights=iter_model.weights, iterations=1000, step=1e-4)
        y_pred = iter_model.predict(test_data.map(lambda row: row.features)).map(lambda row: int(row)).collect()
        rmse = math.sqrt(sum((x - y) ** 2 for x, y in zip(y_pred, y_true)) / len(y_true))
        print(i, ": ", rmse)


def reg_iter_with_different_thread(training_data, test_data, modelname):
    if modelname.__contains__("LR"):
        model = LinearRegressionWithSGD
    elif modelname.__contains__("RR"):
        model = RidgeRegressionWithSGD
    print("*" * 100)
    for i in range(5):
        num = pow(2, i)
        print("Core number is:", num)
        sc.stop()
        sc = SparkContext("local[{}]".format(num), "CAPSTONE_{}".format(num)).getOrCreate()
        spark = SparkSession.builder.master("local[{}]".format(num)).appName("CAPSTONE_{}".format(num)).getOrCreate()
        start = datetime.datetime.now()
        full_model = model.train(training_data, iterations=1000, step=1e-4, miniBatchFraction=1)
        print("full data: ", datetime.datetime.now() - start)
        training_data, add_data = training_data.randomSplit([0.99, 0.01])
        start = datetime.datetime.now()
        add_model = model.train(training_data, iterations=1000, step=1e-4, miniBatchFraction=1)
        print("99% data: ", datetime.datetime.now() - start)
        start = datetime.datetime.now()
        add_model = model.train(add_data, initialWeights=add_model.weights, iterations=1000, step=1e-4)
        print("1% Data: ", datetime.datetime.now() - start)
        print("*" * 100)
