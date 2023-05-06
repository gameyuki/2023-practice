from sklearn.datasets import make_blobs
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

X,y = make_blobs(
    random_state=0,
    n_features=2,
    centers=2,
    cluster_std=1,
    n_samples=300)

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0)

df = pd.DataFrame(X_train)
df["target"] = y_train

df0 = df[df["target"] == 0]
df1 = df[df["target"] == 1]
plt.figure(figsize=(5, 5))

plt.scatter(df0[0], df0[1], color="b", alpha=0.5)
plt.scatter(df1[0], df1[1], color="r", alpha=0.5)
plt.title("title:75%")
plt.show()

df = pd.DataFrame(X_test)
df["target"] = y_test

df0 = df[df["target"] == 0]
df1 = df[df["target"] == 1]
plt.figure(figsize=(5, 5))

plt.scatter(df0[0], df0[1], color="b", alpha=0.5)
plt.scatter(df1[0], df1[1], color="r", alpha=0.5)
plt.title("test:25%")
plt.show()

