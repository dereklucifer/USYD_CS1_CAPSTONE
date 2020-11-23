import pyspark
from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark.sql.functions import isnull, when, count, col
from pyspark.sql.functions import col, isnan, when, trim
from pyspark.ml.feature import StringIndexer

from pyspark.python.pyspark.shell import sc
from pyspark.mllib.regression import LabeledPoint

from pyspark.conf import SparkConf

from pyspark.mllib.regression import LinearRegressionWithSGD, RidgeRegressionWithSGD
from pyspark.mllib.classification import SVMWithSGD, LogisticRegressionWithSGD, LogisticRegressionWithLBFGS
from pyspark.mllib.clustering import KMeans

import datetime
import math
import numpy as np

from  classificaiton import *
from  regression import *
from cluster import *
from data_preprocessing import *

'''
model name:
Classification_LR
Classification_SVM
Classification_BFGS
Classification_KMeans
Regression_LR
Regression_RR

Dataset:
TORQUE
HIGGS
SUSY
EMNIST_0-1
EMNIST_BY_CLASS
EMNIST_BY_DIGITS
'''

# initial environment
sc = SparkContext("local[*]", "TORQUE_Ridge_Data").getOrCreate()
spark = SparkSession.builder.master("local[*]").appName('CAPSTONE').getOrCreate()


def validate_performance(spark,modelname, dataset):
    training_data, test_data = processing_data(spark,modelname, dataset=dataset)
    num_of_class = 2
    if dataset.__contains__('CLASS'):
        num_of_class = 62
    elif dataset.__contains__('DIGITS'):
        num_of_class = 10
    # compare static model and retrain model
    # test result include training accuracy/RMSE and training time
    # static model training use 100% data
    # retrain model training use 99% data and 1% extral data
    if modelname.__contains__('KMeans'):
        clu_compare_with_static_model(training_data, test_data, modelname)
        clu_static_with_different_data_size(training_data, test_data, modelname)
        clu_iter_with_different_training_time(training_data, test_data, modelname)
        clu_iter_with_different_training_size(training_data, test_data, modelname)
        clu_iter_with_different_thread(training_data, test_data, modelname)

    elif modelname.__contains__('Classification'):
        cls_compare_with_static_model(training_data, test_data, modelname, num_of_class=num_of_class)
        cls_static_with_different_data_size(training_data, test_data, modelname, num_of_class=num_of_class)
        cls_iter_with_different_training_time(training_data, test_data, modelname, num_of_class=num_of_class)
        cls_iter_with_different_training_size(training_data, test_data, modelname, num_of_class=num_of_class)
        cls_iter_with_different_thread(training_data, test_data, modelname, num_of_class=num_of_class)

    elif modelname.__contains__('Regression'):
        reg_compare_with_static_model(training_data, test_data, modelname)
        reg_static_with_different_data_size(training_data, test_data, modelname)
        reg_iter_with_different_training_time(training_data, test_data, modelname)
        reg_iter_with_different_training_size(training_data, test_data, modelname)
        reg_iter_with_different_thread(training_data, test_data, modelname)
    else:
        print('invilid model name')

# Sample
validate_performance(spark,'Classification_LR','EMNIST_0-1')