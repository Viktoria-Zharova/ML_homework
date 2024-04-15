from sklearn import datasets
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

#загрузка данных
iris_data = datasets.load_iris()
X = iris_data.data

#массив значений инерции
inertia_values = []

#используем разное кол-во кластеров , заносим значение инерции
for num_clusters in range(1, 11):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(X)
    inertia_values.append(kmeans.inertia_)

#построение графика локтя
plt.plot(range(1, 11), inertia_values, marker='o')
plt.xlabel('кол-во кластеров')
plt.ylabel('инерция')
plt.title('Elbow Method')
plt.show()
