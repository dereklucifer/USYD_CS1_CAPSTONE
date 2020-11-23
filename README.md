# USYD_CS1_CAPSTONE

## How to run

`main==> #sample`

change two paramater which are `model name` and `dataset name`

## Option

| model name            | dataset                               | filename                                                     |
| --------------------- | ------------------------------------- | ------------------------------------------------------------ |
| Classification_LR     | HIGGS<br />SUSY<br />EMNIST_0-1       | HIGGS.csv<br />SUSY.csv<br />emnist_0_1_train.csv            |
| Classification_SVM    | HIGGS<br />SUSY<br />EMNIST_0-1       | HIGGS.csv<br />SUSY.csv<br />emnist_0_1_train.csv            |
| Classification_BFGS   | EMNIST_BY_CLASS<br />EMNIST_BY_DIGITS | emnist-byclass-train.csv<br />emnist-digits-train.csv        |
| Classification_KMeans | EMNIST_BY_CLASS<br />EMNIST_BY_DIGITS | emnist-byclass-train.csv<br />emnist-digits-train.csv        |
| Regression_LR         | TORQUE                                | Time_series_120rpm.csv<br />Time_series_2000rpm.csv<br />Torque_Table.csv |
| Regression_LR         | TORQUE                                | Time_series_120rpm.csv<br />Time_series_2000rpm.csv<br />Torque_Table.csv |

## Experiment

1. Compare the accuracy and training time of static model and retrain model. The static model uses all data to train only once. The retrain model is trained twice, the first time using 99% data, and the second time using 1% data.

2. Test the impact of different data size on the training time and accuracy of the static model.

3. Train 1% of the data each time, and compare the impact of different training times on retrain model accuracy.

4. The total amount of training remains unchanged. By changing the number of training times and the amount of training data in every step, compare the impact of the asynchronous length on the accuracy.

   e.g. 1% data training 100 times, 2% data training 50 times.

5. Compare the impact of different thread numbers on training time, compare static model and retrain model.

