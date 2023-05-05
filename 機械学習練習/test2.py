from sklearn.datasets import make_blobs
import pandas as pd
import matplotlib.pyplot as plt

X,y = make_blobs(
    random_state=0,
    n_features=2,
    centers=2,
    cluster_std=1,
    n_samples=300)

df = pd.DataFrame(X)
df["target"] = y
print(df.head())

df0 = df[df["target"] == 0]
df1 = df[df["target"] == 1]

plt.figure(figsize=(5, 5))
plt.scatter(df0[0], df0[1], color="b", alpha=0.5)
plt.scatter(df1[0], df1[1], color="r", alpha=0.5)
plt.show()
