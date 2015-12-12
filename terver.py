__author__ = 'dalexiv'

import pandas as pd
import numpy as np
import math

start = 201
finish = 600
n = finish - start + 1
table = [1.65, 1.70, 1.75, 1.81, 1.88, 1.96, 2.06, 2.1, 2.33, 2.58, 3.8]
main_value_from_table = table[5]
MAX = 100000000


def getVariance(series):
    sum = 0
    for value in series:
        sum += math.pow(value - series.mean(), 2)
    return sum / (len(series) - 1)


def doFirstTask():
    print("First task:")

    more_than_year_df = my_df[my_df['SearchTime'] == 5]
    more_than_year_df.to_csv("output_data/output_data_my.csv", sep=',', encoding='utf-8')
    perc = float(more_than_year_df.shape[0]) / n

    left_border = perc - main_value_from_table * math.sqrt((perc * (1 - perc)) / n)
    print("Left border of confidence interval " + str(left_border))

    right_border = perc + main_value_from_table * math.sqrt((perc * (1 - perc)) / n)
    print("Right border of confidence interval " + str(right_border))

    print("")



def countBordersVariance(table_value, series):
    left = (len(series) - 1) * getVariance(series) / (0.5 * math.pow((table_value + math.sqrt(2 * (len(series) - 1))), 2))
    right = (len(series) - 1) * getVariance(series) / (0.5 * math.pow((-table_value + math.sqrt(2 * (len(series) - 1))), 2))
    return left, right


def countBordersMean(series):
    left = series.mean() - main_value_from_table * math.sqrt(getVariance(series) / len(series))
    right = series.mean() + main_value_from_table * math.sqrt(getVariance(series) / len(series))
    return left, right


def doSecondTask():
    print("Second task:")

    known_value_df = my_df[my_df["Benefits"].notnull()]
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
    known_value_df.to_csv("output_data/output_data_my_misha.csv", sep=',', encoding='utf-8')
    less_than_half_df = known_value_df[my_df['SearchTime'] <= 3]

    min_dist = MAX
    cached_left = 0
    cached_right = 0
    cached_i = 0

    for i, table_value in enumerate(table):
        temp_left, temp_right = countBordersVariance(table_value, less_than_half_df["Benefits"])
        cur = temp_right - temp_left

        if (cur < min_dist):
            cached_i = i
            min_dist = cur
            cached_left = temp_left
            cached_right = temp_right

    print("Best confidence interval is from " + str(cached_i) + " to " + str(cached_i + 90))
    print("Left border of confidence interval for distribution " + str(cached_left))
    print("Right border of confidence interval for distribution " + str(cached_right))
    print("Minimum distance is " + str(min_dist))
    print("")



train_df = pd.read_csv("input_data/input_data.csv")
my_df = train_df[(train_df.Obs >= start) & (train_df.Obs <= finish)]

doFirstTask()
doSecondTask()
doThirdTask()
