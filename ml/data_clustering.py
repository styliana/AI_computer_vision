from sklearn.cluster import KMeans  # Using KMeans as originally imported
import matplotlib.pyplot as plt
import seaborn as sns

iris = sns.load_dataset('iris')
X_iris = iris.drop('species', axis=1)
y_iris = iris['species']

model = KMeans(n_clusters=3)  # Using KMeans

model.fit(X_iris)

y_predict = model.predict(X_iris)

plt.scatter(X_iris['sepal_length'], X_iris['sepal_width'], c=y_predict)
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('KMeans Clustering')
plt.show()