import keras
import tensorflow
import sklearn
from sklearn import model_selection, linear_model
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")


names = ["Sample_%i" % i for i in range(188)]
names[187] = "Category"
df = pd.read_csv("train_test_ecg\\mitbih_train.csv", names=names)

print("hello world")
