from pyspark.sql.functions import col, isnan, when, trim
from pyspark.mllib.regression import LabeledPoint


# Processing data
def processing_regression_data(filepath,spark):
    # read data
    rpm_120 = (spark.read.format("csv").option("header", "true").load("./TORQUE/Time_series_120rpm.csv"))
    rpm_2000 = (spark.read.format("csv").option("header", "true").load("./TORQUE/Time_series_2000rpm.csv"))
    Table = (spark.read.format("csv").option("header", "true").load("./TORQUE/Torque_Table.csv"))
    # change data type
    columns_list = rpm_120.columns
    rpm_120 = rpm_120.select([col('`{}`'.format(c)).cast('float').alias(c) for c in columns_list])
    rpm_2000 = rpm_2000.select([col('`{}`'.format(c)).cast('float').alias(c) for c in columns_list])
    # change table name
    Table = Table.select(col('OP').cast('int'), col('Speed in 1/min').cast('int').alias("Speed"),
                         col('T in Nm').alias("T"))
    # join tabel
    result1 = Table.join(rpm_120, (Table.OP == rpm_120.OP) & (Table.Speed == 120)).drop(rpm_120.OP)
    result2 = Table.join(rpm_2000, (Table.OP == rpm_2000.OP) & (Table.Speed == 2000)).drop(rpm_2000.OP)
    data = result1.union(result2)
    data = data.drop('s_n').drop('Speed').drop('OP')

    # remove null value
    def to_null(c):
        return when(~(col(c).isNull() | isnan(col(c)) | (trim(col(c)) == "")), col(c))

    df = data.select([to_null(c).alias(c) for c in data.columns]).na.drop()
    return df


def processing_classification_data(dataset,spark):
    if dataset == 'SUSY' or dataset == 'HIGGS':
        df = (spark.read.format("csv").option("header", "true").load(dataset + ".csv"))
    elif dataset == 'EMNIST_0-1':
        df = (spark.read.format("csv").option("header", "true").load("./EMNIST/EMNIST_0_1.csv"))
    elif dataset == 'EMNIST_BY_CLASS':
        df = (spark.read.format("csv").option("header", "true").load("./EMNIST/emnist-byclass-train.csv"))
    elif dataset == 'EMNIST_BY_DIGITS':
        df = (spark.read.format("csv").option("header", "true").load("./EMNIST/emnist-digits-train.csv"))
    else:
        print('Invalid file name')
        return None
    columns_list = df.columns

    df = df.select([col('`{}`'.format(c)).cast('float').alias(c) for c in columns_list])
    return df


def processing_data(spark,modeltype, dataset=""):
    if modeltype.__contains__('regression'):
        df = processing_regression_data(spark)
    else:
        df = processing_classification_data(dataset,spark)
    labelpointRDD = df.rdd.map(lambda row: LabeledPoint(row[1], row[2:]))
    training_data, test_data = labelpointRDD.randomSplit([0.8, 0.2])
    return training_data, test_data
