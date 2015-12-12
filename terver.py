__author__ = 'dalexiv'

import pandas as pd
import numpy as np
import math

start = 1801 # First value of dataset
finish = 2200 # First value of dataset
n = finish - start + 1 # Number of entries in dataset

# Values from table
table_norm_left = [1.65, 1.70, 1.75, 1.81, 1.88, 1.96, 2.06, 2.1, 2.33, 2.58, 3.8]
table_norm_right = list(table_norm_left) # Values reversed
table_norm_right.reverse()

main_value_from_table = 5 # Value for distribution in the second task
table_student = 1.9791241 # Value for mean in the second task
MAX = 100000000 # Just max value

# Counts simple series variance
def getVariance(series):
    sum = 0
    for value in series:
        sum += math.pow(value - series.mean(), 2)
    return sum / (len(series) - 1)


def doFirstTask():
    print("First task:")

    # Creating new dataset which contains only people who wait more than year
    more_than_year_df = my_df[my_df['SearchTime'] == 5]
    more_than_year_df.to_csv("output_data/my_data_1_task.csv", sep=',', encoding='utf-8')
    perc = float(more_than_year_df.shape[0]) / n # These people portion

    left_border = perc - table_norm_left[main_value_from_table] * math.sqrt((perc * (1 - perc)) / n)
    print("Left border of portion confidence interval " + str(left_border))

    right_border = perc + table_norm_left[main_value_from_table] * math.sqrt((perc * (1 - perc)) / n)
    print("Right border of portion confidence interval " + str(right_border))

    print("")

# Counts borders for confidence interval for variance
# with given percentage from the table
def countBordersVariance(i, series):
    local_n = len(series) - 1
    left = local_n * getVariance(series) / (0.5 * math.pow((table_norm_left[i] + math.sqrt(2 * local_n - 1)), 2))
    right = local_n * getVariance(series) / (0.5 * math.pow((-table_norm_right[i] + math.sqrt(2 * local_n - 1)), 2))
    return left, right

# Counts borders for confidence interval for mean with 90%
def countBordersMean(series):
    left = series.mean() - table_student * math.sqrt(getVariance(series) / len(series))
    right = series.mean() + table_student * math.sqrt(getVariance(series) / len(series))
    return left, right


def doSecondTask():
    print("Second task:")

    # Choosing only not null values
    known_value_df = my_df[my_df["Benefits"].notnull()]
    known_value_df.to_csv("output_data/my_data_2_task.csv", sep=',', encoding='utf-8')

    left_mean, right_mean = countBordersMean(known_value_df["Benefits"])

    print("Mean for dataset is " + str(known_value_df["Benefits"].mean()))
    print("Variance for dataset is " + str(getVariance(known_value_df["Benefits"])))

    print("Left border of confidence interval for mean " + str(left_mean))
    print("Right border of confidence interval for mean " + str(right_mean))

    left_dist, right_dist = countBordersVariance(main_value_from_table, known_value_df["Benefits"])

    print("Left border of confidence interval for variance " + str(left_dist))
    print("Right border of confidence interval for variance " + str(right_dist))

    print("")


def doThirdTask():
    print("Third task:")
    # Do the preprocessing
    known_value_df = my_df[my_df["Benefits"].notnull()]
    less_than_half_df = known_value_df[known_value_df['SearchTime'] <= 3]
    less_than_half_df.to_csv("output_data/my_data_3_task.csv", sep=',', encoding='utf-8')

    best_i = 0
    min_dist = MAX
    best_left = 0
    best_right = 0
    # Running over all possible situations
    for i in range(1, len(table_norm_left) - 1):
        temp_left, temp_right = countBordersVariance(i, less_than_half_df["Benefits"])
        cur = temp_right - temp_left

        if (cur < min_dist):
            best_i = i
            min_dist = cur
            best_left = temp_left
            best_right = temp_right

    print("Best confidence interval is from " + str(best_i) + " to " + str(best_i + 90))
    print("Left border of confidence interval for distribution " + str(best_left))
    print("Right border of confidence interval for distribution " + str(best_right))
    print("Minimum distance is " + str(min_dist))
    print("")


train_df = pd.read_csv("input_data/input_data.csv")
# Only my vaues in dataset
my_df = train_df[(train_df.Obs >= start) & (train_df.Obs <= finish)]

doFirstTask()
doSecondTask()
doThirdTask()
