import numpy as np
import matplotlib.pyplot as plt
import imageio
from sklearn.datasets import load_iris

#загрузка данных
iris_data = load_iris()
X = iris_data.data
num_clusters = 3  #число кластеров
np.random.seed(0)
initial_centroids = X[np.random.choice(range(len(X)), num_clusters, replace=False)]  #инициализация начальных центроидов

#считаем расстояния между двумя точками
def calculate_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

#назначаем точки
def assign_clusters(X, centroids):
    clusters = []
    for point in X:
        distances = [calculate_distance(point, centroid) for centroid in centroids]
        cluster = np.argmin(distances)
        clusters.append(cluster)
    return np.array(clusters)

#обновляем центроиды
def update_centroids(X, clusters, num_clusters):
    centroids = []
    for i in range(num_clusters):
        cluster_points = X[clusters == i]
        centroid = np.mean(cluster_points, axis=0)
        centroids.append(centroid)
    return np.array(centroids)

max_iterations = 100  #макс кол-во итераций
images = []

#kmeans
for iteration in range(max_iterations):
    clusters = assign_clusters(X, initial_centroids)  #назнач точки кластерам

    #отображение состояния кластеризации в данном моменте
    plt.figure(figsize=(8, 6))
    colors = ['r', 'g', 'b']
    for i in range(num_clusters):
        cluster_points = X[clusters == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=colors[i], label=f'Cluster {i + 1}')
    plt.scatter(initial_centroids[:, 0], initial_centroids[:, 1], c='black', marker='x', label='Centroids')
    plt.xlabel('Sepal Length')
    plt.ylabel('Sepal Width')
    plt.title(f'K-means Clustering - Iteration {iteration + 1}')
    plt.legend()

    plt.savefig(f'iteration_{iteration}.png')  #сохраняем изображение графика
    plt.close()

    images.append(imageio.imread(f'iteration_{iteration}.png'))  #добавляем изображение в список для создания гифки

    new_centroids = update_centroids(X, clusters, num_clusters)  #обновляем центроиды

    #проыеряем сходимость алгоритма
    if np.all(initial_centroids == new_centroids):
        print("Algorithm converged at iteration", iteration + 1)
        break

    initial_centroids = new_centroids

#создаем гифку
imageio.mimsave('kmeans_animation.gif', images, fps=2)
