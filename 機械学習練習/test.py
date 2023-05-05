from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt

iris = datasets.load_iris()
df = pd.DataFrame(iris.data)
df.columns = iris.feature_names
df['target'] = iris.target

df0 = df[df["target"] == 0]
df1 = df[df["target"] == 1]
df2 = df[df["target"] == 2]

plt.figure(figsize=(5, 5))
xx = "sepal width (cm)"
df0[xx].hist(color ="b",alpha=0.5)
df1[xx].hist(color ="r",alpha=0.5)
df2[xx].hist(color ="g",alpha=0.5)

plt.xlabel(xx)
plt.show()




