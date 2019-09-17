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

predict = "Category"
x = np.array(df.drop([predict], 1))
y = np.array(df[predict])

best = 0
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.1)

model = KNeighborsClassifier(n_neighbors=7)
model.fit(x_train, y_train)
acc = model.score(x_test, y_test)

predictions = model.predict(x_test)
for x in range(len(predictions)):
    print(predictions[x], y_test[x])

print(acc)

