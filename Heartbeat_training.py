# DAVID DING
# SEPTEMBER 17TH 2019       UPDATED SEPTEMBER 19th
# WORK IN PROGRESS
# This file creates the machine learning model for identifying HR from ECG Data
# Currently using KNNClassifyer from SKLearn
# TODO: Experiement with Keras Sequential models for better accuracy


# ======================================== IMPORTS ========================================
import keras
import tensorflow
import sklearn
from sklearn import model_selection, linear_model
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
style.use("ggplot")


# ======================================== VARIABLES ========================================
# names (list) is the column for sample names, used for indexing (of course)
# df_n (pd.DataFrame) for n is any number of imported CSV files
# df (pd.DataFrame) is the overarching df used for the model_selection.train_test_split function
# predict (str) is the column name for which variable to predict
# X (list) list of values that act as the input
# y (list) list of values that act as output
# X_train, X_test, Y_train, Y_test (list) are all values used for splitting and training the model from the df database
# best (int) represents the best accuracy of the model
# best_model (sklearn.neighbors.KNeighborsClassifier) is used to store the best model to be pickled
# model (sklearn.neighbors.KNeighborsClassifier) is the current model, to be overwritten every instance of for-loop
names = ["Sample_%i" % i for i in range(151)]
names[150] = "isHR"
df_1 = pd.read_csv("TESTS/TEST.csv", names=names)
df_2 = pd.read_csv("TESTS/TEST2.csv", names=names)
df_3 = pd.read_csv("TESTS/TEST3.csv", names=names)
df_4 = pd.read_csv("TESTS/TEST4.csv", names=names)
df = pd.concat([df_1[1:], df_2[1:], df_3[1:], df_4[1:]])

predict = "isHR"
X = np.array(df.drop([predict], 1))
y = np.array(df[predict])

best = 0
best_model = None
model = None

# Train the model 50 times to generate new randomly split data for training and testing purposes
for i in range(50):

    # Split, Train, Test, then fit and evaluate the model
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
    model = KNeighborsClassifier(n_neighbors=7)
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)

    # Save the best model
    if acc > best:
        best = acc
        best_model = model

# Output
print("Best accuracy from the trials was: %.2f" % best)

# Save the best model for easy extraction later
pickle.dump(best_model, open("pickled_models/model_all.pickle", "wb"))
