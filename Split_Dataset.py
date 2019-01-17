#!/usr/bin/env python3
# orient.py : Dataset splitter
# R. Thakkar, 2018

import pandas as pd
import random
import math


def dataset_split(file, split, seed):
    train_data = pd.read_table(file, sep=" ", header=None)
    train_data_values = train_data.values
    random.Random(seed).shuffle(train_data_values)
    desired_indices = math.ceil(len(train_data_values) * split)
    desired_data = train_data_values[:desired_indices]
    desired_data_df = pd.DataFrame(desired_data)
    print("Generating "+str(split*100)+ "% dataset.....")
    desired_data_df.to_csv("train-data-" + str(split) + ".txt", sep=' ', index=False, header=False)


dataset_split("train-data.txt", split=0.9, seed=90)
dataset_split("train-data.txt", split=0.8, seed=80)
dataset_split("train-data.txt", split=0.7, seed=70)
dataset_split("train-data.txt", split=0.6, seed=60)
dataset_split("train-data.txt", split=0.5, seed=50)
dataset_split("train-data.txt", split=0.4, seed=40)
dataset_split("train-data.txt", split=0.3, seed=30)
dataset_split("train-data.txt", split=0.2, seed=20)
dataset_split("train-data.txt", split=0.1, seed=10)
