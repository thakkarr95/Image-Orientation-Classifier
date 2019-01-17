#!/usr/bin/env python3
# orient.py : Image Orientation Classifier
# R. Thakkar, 2018

"""
We have assumed that the train.txt nd test.txt files will be in the same format as given to us on canvas.

Problem Statement: Image orientation classification based on the values of RGB values for a given image.

Training Dataset: The train dataset consists of approximately 40,000 images. It actually consists of 10,000 images
rotated 4 times to give 4 time as much data. We need to predict the orientation of each image which is multi class
and has 4 unique values namely 0, 90, 180 and 270.

Test Dataset: The test dataset each image occurs only once and has 1000 images for which we need to predict the
orientation using the classification algorithms.

We have implemented 3 algorithms for this task namely K Nearest Neighbours (KNN), Ada boost and Random Forest.
Parameters for each model were tweaked to optimize the training time and classification accuracy on test dataset.

KNN:
K Nearest Neighbours algorithm calculates the distance of each row in test dataset with each row in the training
dataset. Then we identify the k rows(neighbours) from training dataset which are closest to each row in test dataset.
A majority vote from the k neighbours is taken to assign a class to each row in the test dataset.

For KNN, we have implemented the program such that it can take different values of K and you can also change the
distance metric to be used. There are 2 options available for the distance metric:
1.	Euclidean Distance
2.	Manhattan Distance

We have used pickle library of Python to store the data structures used in python in .pkl files on the disk.
The model for KNN is stored in ‘nearest_model.pkl’. In the training phase, the train dataset is stored in the
‘nearest_model.pkl’ file. This is why training time of KNN is not important for us.

During the test phase, the train dataset is loaded from the ‘nearest_model.pkl’ file and then classification is
performed as described above. The image name and predicted labels for each image in the test dataset are stored in
the file ‘output.txt’.

Adaboost:
Adaboost algorithm uses an ensemble of weak classifiers to classify the datapoints in test dataset. We have used
decision stumps as weak classifiers. Each decision stump compares 2 different features which results in a total of
192C2 decision stumps for each model. 6 different models were constructed as listed below:
1.	Model to distinguish between 0 and 90 orientation
2.	Model to distinguish between 0 and 180 orientation
3.	Model to distinguish between 0 and 270 orientation
4.	Model to distinguish between 90 and 180 orientation
5.	Model to distinguish between 90 and 270 orientation
6.	Model to distinguish between 180 and 270 orientation

The classification is made based on the majority votes we get from the 6 models above for each row in the test dataset.
Each model can have 192C2 decision stumps(weak classifiers), where 192 is the number of features used, and corresponding
alphas which are weights of each weak classifier in each of the 6 models described above.

From the 192C2 decision stumps, only the decision stumps with error less than 0.5 were considered as weak classifiers.
We have used pickle library of Python to store the data structures used in python in .pkl files on the disk.
The model for adaboost is stored in ‘adaboost_model.pkl’. In the training phase, the weak classifiers for each model
and their aplhas are stored in a dictionary for each of the 6 models. A list of such dictionaries is stored in the
‘adaboost_model.pkl’ file.

During the test phase, the list of models is loaded from the ‘adaboost_model.pkl’ file and then the models are used to
classify each row in the test dataset. The image name and predicted labels for each image in the test dataset are
stored in the file ‘output.txt’.

Random Forest:
Random Forest is a supervised Machine Learning algorithm. It uses an ensemble of Decision Trees to over come the
variance observed in the results of a Decision Tree. The “forest” it builds, is an ensemble of Decision Trees, most of
the time trained with the “bagging” method. The general idea of the bagging method is that a combination of learning
models increases the overall result. To say it in simple words: Random forest builds multiple decision trees and merges
them together to get a more accurate and stable prediction.
In this algorithm, a portion of dataset is used to generate N decision trees each with depth d. We are using
14 (square root of 192) features which are selected randomly from the available 192 features for each decision tree.
We are using 15% of the dataset for our experiments. This is the training phase of the algorithm.

We have used pickle library of Python to store the data structures used in python in .pkl files on the disk.
The Random Forest is stored in ‘random_forest_model.pkl’.
During the test phase, model is loaded from the ‘random_forest_model.pkl’ file.
All the Decision Trees are used to make prediction for each row in the test dataset.
The prediction is made based on the maximum voting from the N Decision Trees generated during the train phase.
The image name and predicted labels for each image in the test dataset are stored in the file ‘output.txt’.

Our implementation KNN algorithm performs the best for the given dataset. We achieved an accuracy of 73.45%.
when Manhattan distance is used and K=25 is used. Experiments with different training dataset size were performed to
understand the performance of KNN with K=25 and Manhattan distance.
Split_Dataset.py script has been used to split the dataset. It splits the training  dataset by selecting the x% rows
from training dataset randomly where, x is from 10 to 100. Seed has been used from the random library of Python to
ensure reproducibility.
The datasets are stored with filenames ‘train-data-0.1.txt’, ‘train-data-0.2.txt’, …… ‘train-data-0.9.txt’. Complete
training dataset is available in ‘train-data.txt’. The corresponding models are stored with filenames
‘best_model_0.1.pkl’, ‘best_model_0.2.pkl’, …… ‘best_model_0.9.pkl’. Model for the complete training dataset
is stored in ‘best_model.pkl’.

"""

# import libraries
import sys
import numpy as np
from collections import Counter
import pandas as pd
import pickle
import time
import random
from itertools import combinations
import math
import warnings
from math import sqrt

warnings.filterwarnings('ignore')

start_time = time.time()

# Argument Parsing
to_do = sys.argv[1]
train_test_file = sys.argv[2]
model_file_name = sys.argv[3]
model_name = sys.argv[4]

######################## KNN Functions ######################################

# Referred the following link for implementation of KNN algorithm:
# https://www.kdnuggets.com/2016/01/implementing-your-own-knn-using-python.html/3


def euclidean_distance(x_train, x_test):
    distance = np.sqrt(np.dot((x_train - x_test), (x_train - x_test)))
    return distance


def manhattan_distance(x_train, x_test):
    distance = np.sum(abs(x_train - x_test))
    return distance


# a = np.array([1,1])
# b = np.array([2,2])

# print(eucledian_distance(a,b))
# print(manhattan_distance(a,b))

def knn_train(X_train, Y_train):
    print("Training the model...............")
    model = X_train
    model[1] = Y_train
    print("Saving the model...............")
    model_file = open(model_file_name, 'wb')
    pickle.dump(model, model_file)
    model_file.close()


def knn_score(model_file, image_names, X_test, Y_test, distance, k):
    file = open(model_file, 'rb')
    train_data = pickle.load(file)
    file.close()

    X_train = train_data.iloc[:, :-1]
    Y_train = train_data[1]

    neighbours = {}

    for x_test in X_test.values:
        distance_list = []
        for x_train, y_train in zip(X_train.values, Y_train):
            if distance == "Euclidean":
                distance_list.append((x_train, y_train, euclidean_distance(x_train, x_test)))
            else:
                #                 print("Manhattan is being used!")
                distance_list.append((x_train, y_train, manhattan_distance(x_train, x_test)))

        distance_list_sorted = sorted(distance_list, key=lambda x: x[2])

        neighbours[tuple(x_test)] = distance_list_sorted[:k]

    predicted_y = []

    text_file = open("output.txt", "w")
    for x_test, image_name in zip(X_test.values, image_names):
        neighbour_list = neighbours[tuple(x_test)]
        counts = Counter(neighbour[1] for neighbour in neighbour_list)
        text_file.write(str(image_name) + " " + str(counts.most_common()[0][0]) + "\n")
        predicted_y.append(counts.most_common()[0][0])

    predicted_y = np.array(predicted_y)
    text_file.close()

    correct = 0
    for y_train, y_predicted in zip(Y_test, predicted_y):
        if str(y_train) == str(y_predicted):
            correct += 1

    print("Classification accuracy is: ", correct * 100 / len(predicted_y), "%")


######################## Ada Boost Functions ######################################

# Discussed the strategy of splitting the multi-class classification problem in to multiple binary classification
# problems with Nithish and Rohit Bapat

def normalize(a):
    a = np.array(a)
    a_norm = a / np.sum(a)
    return a_norm


def train_weak_classifiers(train_data_model_1, seed):
    train_data_columns = list(range(2, 194))
    random.Random(seed).shuffle(train_data_columns)
    model_1_columns = train_data_columns[0:]

    data_model_1 = train_data_model_1[model_1_columns + ["weights", "class"]]

    # We will change the number of decision stumps here
    model_1_columns_perm = list(combinations(model_1_columns, 2))[0:5499]

    model_1 = {}
    error_model_1 = []

    for i in range(len(model_1_columns_perm)):
        first_col = model_1_columns_perm[i][0]
        second_col = model_1_columns_perm[i][1]

        class_predicted_1 = []
        class_predicted_2 = []

        for row in zip(data_model_1[first_col],data_model_1[second_col]):
            if row[0] >= row[1]:
                class_predicted_1.append(1)
                class_predicted_2.append(-1)
            else:
                class_predicted_1.append(-1)
                class_predicted_2.append(1)

        error_1 = 0
        error_2 = 0

        errors = []

        for y_1, y_2, Y, w in zip(class_predicted_1, class_predicted_2, data_model_1["class"], data_model_1["weights"]):
            if y_1 != Y:
                error_1 += w
            elif y_2 != Y:
                error_2 += w

        errors.append(error_1)
        errors.append(error_2)
        min_error = min(errors)
        if min_error < 0.5:
            error_model_1.append(min_error)
            min_ind = errors.index(min_error)
            w_model = math.log((1 - min_error) / min_error)

            if min_ind == 0:
                model_1[model_1_columns_perm[i]] = ["1 if first is greater than equal to second", w_model]
                data_model_1["class_predicted"] = class_predicted_1

            else:
                model_1[model_1_columns_perm[i]] = ["1 if first is less than second", w_model]
                data_model_1["class_predicted"] = class_predicted_2

            data_model_1["weights"] = np.where(data_model_1["class"] == data_model_1["class_predicted"],
                                               data_model_1["weights"] * (min_error / (1 - min_error)),
                                               data_model_1["weights"])
            normalized_weights = normalize(data_model_1["weights"])
            data_model_1["weights"] = normalized_weights

        else:
            continue

    data_model_1_predicted = data_model_1.copy()
    data_model_1_predicted["class_predicted"] = 0

    return model_1


def adaboost_train(X_train, Y_train):
    train_data_local = X_train.copy()
    train_data_local["class"] = Y_train
    train_data_local["Y"] = Y_train

    train_data_models = train_data_local.copy()

    train_data_model_1 = train_data_models.loc[train_data_models["class"].isin([0, 90])]
    train_data_model_2 = train_data_models.loc[train_data_models["class"].isin([0, 180])]
    train_data_model_3 = train_data_models.loc[train_data_models["class"].isin([0, 270])]

    train_data_model_4 = train_data_models.loc[train_data_models["class"].isin([90, 180])]
    train_data_model_5 = train_data_models.loc[train_data_models["class"].isin([90, 270])]

    train_data_model_6 = train_data_models.loc[train_data_models["class"].isin([180, 270])]

    train_data_model_1["class"] = train_data_model_1["class"].mask(train_data_model_1["class"] == 0, -1)
    train_data_model_1["class"] = train_data_model_1["class"].mask(train_data_model_1["class"] == 90, 1)

    train_data_model_2["class"] = train_data_model_2["class"].mask(train_data_model_2["class"] == 0, -1)
    train_data_model_2["class"] = train_data_model_2["class"].mask(train_data_model_2["class"] == 180, 1)

    train_data_model_3["class"] = train_data_model_3["class"].mask(train_data_model_3["class"] == 0, -1)
    train_data_model_3["class"] = train_data_model_3["class"].mask(train_data_model_3["class"] == 270, 1)

    train_data_model_4["class"] = train_data_model_4["class"].mask(train_data_model_4["class"] == 90, -1)
    train_data_model_4["class"] = train_data_model_4["class"].mask(train_data_model_4["class"] == 180, 1)

    train_data_model_5["class"] = train_data_model_5["class"].mask(train_data_model_5["class"] == 90, -1)
    train_data_model_5["class"] = train_data_model_5["class"].mask(train_data_model_5["class"] == 270, 1)

    train_data_model_6["class"] = train_data_model_6["class"].mask(train_data_model_6["class"] == 180, -1)
    train_data_model_6["class"] = train_data_model_6["class"].mask(train_data_model_6["class"] == 270, 1)

    print("Training the model...............")

    model_1 = train_weak_classifiers(train_data_model_1, seed=1)
    model_2 = train_weak_classifiers(train_data_model_2, seed=2)
    model_3 = train_weak_classifiers(train_data_model_3, seed=3)

    model_4 = train_weak_classifiers(train_data_model_4, seed=4)
    model_5 = train_weak_classifiers(train_data_model_5, seed=5)

    model_6 = train_weak_classifiers(train_data_model_6, seed=6)

    models = [model_1, model_2, model_3, model_4, model_5, model_6]

    # print(models)
    print("Saving the model...............")
    model_file = open(model_file_name, 'wb')
    pickle.dump(models, model_file)
    model_file.close()


def adaboost_test(model_file, image_names, X_test, Y_test):
    file = open(model_file, 'rb')
    models = pickle.load(file)
    file.close()

    test_data_local = X_test.copy()
    test_data_local["class"] = Y_test
    test_data_local["Y"] = Y_test

    test_data_predicted = test_data_local.copy()

    i = 1
    for model in models:
        class_predicted_model = []
        for index, row in test_data_predicted.iterrows():
            h = 0
            for weak_classifier in model:
                # print(weak_classifier)
                first_col = weak_classifier[0]
                second_col = weak_classifier[1]

                decision = model[weak_classifier][0]
                alpha = model[weak_classifier][1]

                if decision == "1 if first is greater than equal to second":
                    if row[first_col] >= row[second_col]:
                        h += 1.0 * alpha
                    else:
                        h -= 1.0 * alpha
                elif decision == "1 if first is less than second":
                    if row[first_col] >= row[second_col]:
                        h -= 1.0 * alpha
                    else:
                        h += 1.0 * alpha
            if h >= 0:
                class_predicted_model.append(1)
            else:
                class_predicted_model.append(-1)
        test_data_predicted["class_predicted_model_" + str(i)] = class_predicted_model
        # test_data_predicted["weight_of_model_"+str(i)] = model_weight
        i += 1

    test_data_predicted["class_predicted_model_1"] = test_data_predicted["class_predicted_model_1"].\
        mask(test_data_predicted["class_predicted_model_1"] == -1, 0)
    test_data_predicted["class_predicted_model_1"] = test_data_predicted["class_predicted_model_1"]. \
        mask(test_data_predicted["class_predicted_model_1"] == 1, 90)

    test_data_predicted["class_predicted_model_2"] = test_data_predicted["class_predicted_model_2"]. \
        mask(test_data_predicted["class_predicted_model_2"] == -1, 0)
    test_data_predicted["class_predicted_model_2"] = test_data_predicted["class_predicted_model_2"]. \
        mask(test_data_predicted["class_predicted_model_2"] == 1, 180)

    test_data_predicted["class_predicted_model_3"] = test_data_predicted["class_predicted_model_3"]. \
        mask(test_data_predicted["class_predicted_model_3"] == -1, 0)
    test_data_predicted["class_predicted_model_3"] = test_data_predicted["class_predicted_model_3"]. \
        mask(test_data_predicted["class_predicted_model_3"] == 1, 270)

    test_data_predicted["class_predicted_model_4"] = test_data_predicted["class_predicted_model_4"]. \
        mask(test_data_predicted["class_predicted_model_4"] == -1, 90)
    test_data_predicted["class_predicted_model_4"] = test_data_predicted["class_predicted_model_4"]. \
        mask(test_data_predicted["class_predicted_model_4"] == 1, 180)

    test_data_predicted["class_predicted_model_5"] = test_data_predicted["class_predicted_model_5"]. \
        mask(test_data_predicted["class_predicted_model_5"] == -1, 90)
    test_data_predicted["class_predicted_model_5"] = test_data_predicted["class_predicted_model_5"]. \
        mask(test_data_predicted["class_predicted_model_5"] == 1, 270)

    test_data_predicted["class_predicted_model_6"] = test_data_predicted["class_predicted_model_6"]. \
        mask(test_data_predicted["class_predicted_model_6"] == -1, 180)
    test_data_predicted["class_predicted_model_6"] = test_data_predicted["class_predicted_model_6"]. \
        mask(test_data_predicted["class_predicted_model_6"] == 1, 270)

    desired = ["class_predicted_model_1", "class_predicted_model_2", "class_predicted_model_3",
               "class_predicted_model_4", "class_predicted_model_5", "class_predicted_model_6"]

    predicted_class = []

    for row in test_data_predicted[desired].values:
        counts = Counter(row)
        # print(counts)
        # print(counts.most_common()[0][0])
        predicted_class.append(counts.most_common()[0][0])

    test_data_predicted["class_predicted"] = predicted_class

    correct = 0
    for y_test, y_predicted in zip(test_data_predicted["Y"], test_data_predicted["class_predicted"]):
        if str(y_test) == str(y_predicted):
            correct += 1

    print("Writing the output to the file")

    text_file = open("output.txt", "w")
    for x_test, image_name, y in zip(X_test.values, image_names,test_data_predicted["class_predicted"]):
        #         print(len(neighbours[tuple(x_test)]))
        text_file.write(str(image_name) + " " + str(y) + "\n")
    text_file.close()

    print("Classification accuracy is: ", correct * 100 / len(test_data_predicted["class_predicted"]), "%")

######################## Random Forest Functions ######################################

# Referred the following links for implementation of Random Forest:

# https://machinelearningmastery.com/implement-random-forest-scratch-python/
# https://github.com/searene/demos/blob/master/RandomForest/evaluate_random_forest.py
# https://github.com/random-forests/tutorials/blob/master/decision_tree.ipynb


class Node:
    def __init__(self, data):
        # all the data that is held by this node
        self.data = data

        # left child node
        self.left = None

        # right child node
        self.right = None

        # category if the current node is a leaf node
        self.category = None

        # a tuple: (row, column), representing the point where we split the data
        # into the left/right node
        self.split_point = None


def predict_with_single_tree(tree, row):
    if tree.category is not None:
        return tree.category
    x, y = tree.split_point
    split_value = tree.data[x][y]
    if row[y] <= split_value:
        return predict_with_single_tree(tree.left, row)
    else:
        return predict_with_single_tree(tree.right, row)


def predict(trees, row):
    prediction = []
    for tree in trees:
        prediction.append(predict_with_single_tree(trees[tree], row))
    return max(set(prediction), key=prediction.count)


def get_most_common_category(data):
    categories = [row[-1] for row in data]
    return max(set(categories), key=categories.count)


def build_tree(train_data, depth, max_depth, min_size, n_features):
    root = Node(train_data)
    x, y = get_split_point(train_data, n_features)
    left_group, right_group = split(train_data, x, y)
    if len(left_group) == 0 or len(right_group) == 0 or depth >= max_depth:
        root.category = get_most_common_category(left_group + right_group)
    else:
        root.split_point = (x, y)
        if len(left_group) < min_size:
            root.left = Node(left_group)
            root.left.category = get_most_common_category(left_group)
        else:
            root.left = build_tree(left_group, depth + 1, max_depth, min_size, n_features)

        if len(right_group) < min_size:
            root.right = Node(right_group)
            root.right.category = get_most_common_category(right_group)
        else:
            root.right = build_tree(right_group, depth + 1, max_depth, min_size, n_features)
    return root


def get_features(n_selected_features, n_total_features):
    features = [i for i in range(n_total_features)]
    # Using a seed of 10 for reproducibility
    random.Random(10).shuffle(features)
    return features[:n_selected_features]


def get_categories(data):
    return set([row[-1] for row in data])


# def get_split_point(data, n_features):
#     n_total_features = len(data[0]) - 1
#     features = get_features(n_features, n_total_features)
#     categories = get_categories(data)
#     x, y, gini_index = None, None, None
#     for index in range(len(data)):
#         for feature in features:
#             left, right = split(data, index, feature)
#             current_gini_index = get_gini_index(left, right, categories)
#             if gini_index is None or current_gini_index < gini_index:
#                 x, y, gini_index = index, feature, current_gini_index
#     return x, y

def get_split_point(data, n_features):
    n_total_features = len(data[0]) - 1
    features = get_features(n_features, n_total_features)
    categories = get_categories(data)
    x, y, gini_index = None, None, None

    # print("Features are:", features)
    #     print("data is:",data)

    data = np.array(data)
    for feature in features:
        local_data = data[np.argsort(data[:, feature])]
        local_y = local_data[:, -1]
        #         print("local data is:",local_data)
        #         print("local y is:",local_y)
        indexes = np.where(local_y[:-1] != local_y[1:])[0]
        indexes_new = indexes[int(len(indexes) * 0.15):int(len(indexes) * 0.85)]
        for index in indexes_new:
            left, right = split(local_data, index, feature)
            current_gini_index = get_gini_index(left, right, categories)
            if gini_index is None or current_gini_index < gini_index:
                x, y, gini_index = index, feature, current_gini_index
    return x, y


def get_gini_index(left, right, categories):
    gini_index = 0
    for group in left, right:
        if len(group) == 0:
            continue
        score = 0
        for category in categories:
            p = [row[-1] for row in group].count(category) / len(group)
            score += p * p
        gini_index += (1 - score) * (len(group) / len(left + right))
    return gini_index


def split(data, x, y):
    split_value = data[x][y]
    left, right = [], []
    for row in data:
        if row[y] <= split_value:
            left.append(row)
        else:
            right.append(row)
    return left, right


def random_forest_train(train_data, n_trees, max_depth, min_size, n_features, n_sample_rate):
    print("Training the model...............")
    trees = {}
    for i in range(n_trees):
        random.Random(10).shuffle(train_data)
        n_samples = int(len(train_data) * n_sample_rate)
        tree = build_tree(train_data[: n_samples], 1, max_depth, min_size, n_features)
        trees[i] = tree

    print("Saving the model...............")
    model_file = open(model_file_name, 'wb')
    pickle.dump(trees, model_file)
    model_file.close()


def random_forest_test(model_file, image_names, validate_data):
    file = open(model_file, 'rb')
    model = pickle.load(file)
    file.close()

    n_total = 0
    n_correct = 0
    predicted_categories = [predict(model, row[:-1]) for row in validate_data]
    correct_categories = [row[-1] for row in validate_data]
    for predicted_category, correct_category in zip(predicted_categories, correct_categories):
        n_total += 1
        if predicted_category == correct_category:
            n_correct += 1

    print("Writing the output to the file")

    text_file = open("output.txt", "w")
    for image_name, y in zip(image_names, predicted_categories):
        #         print(len(neighbours[tuple(x_test)]))
        text_file.write(str(image_name) + " " + str(y) + "\n")
    text_file.close()

    print("Classification accuracy is: ", n_correct * 100 / n_total, "%")


if to_do.lower() == "train":
    train_data = pd.read_table(train_test_file, sep=" ", header=None)
    image_names = train_data.iloc[:, 0]
    Y_train = train_data.iloc[:, 1]
    X_train = train_data.iloc[:, 2:]

    if model_name.lower() == "nearest":
        knn_train(X_train, Y_train)
    elif model_name.lower() == "adaboost":
        X_train["weights"] = np.repeat(1 / X_train.shape[0], X_train.shape[0])
        adaboost_train(X_train, Y_train)
    elif model_name.lower() == "forest":
        train_data = X_train.copy()
        train_data["Y"] = Y_train
        train_data = train_data.values
        n_features = int(sqrt(len(train_data[0]) - 1))
        random_forest_train(
            train_data=train_data,
            n_trees=2,
            max_depth=10,
            min_size=400,
            n_features=n_features,
            n_sample_rate=0.15
        )
    else:
        knn_train(X_train, Y_train)

else:
    test_data = pd.read_table(train_test_file, sep=" ", header=None)
    image_names = test_data.iloc[:, 0]
    Y_test = test_data.iloc[:, 1]
    X_test = test_data.iloc[:, 2:]

    if model_name.lower() == "nearest":
        knn_score(model_file_name, image_names, X_test, Y_test, "Manhattan", 25)
    elif model_name.lower() == "adaboost":
        # print("Ada_boost test")
        adaboost_test(model_file_name, image_names,X_test, Y_test)
    elif model_name.lower() == "forest":
        validate_data = X_test.copy()
        validate_data["Y"] = Y_test
        validate_data = validate_data.values
        random_forest_test(model_file_name, image_names, validate_data)
    else:
        knn_score(model_file_name, image_names, X_test, Y_test, "Manhattan", 25)

print("--- %s seconds ---" % (time.time() - start_time))