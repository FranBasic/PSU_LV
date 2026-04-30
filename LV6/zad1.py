import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import datasets
import mglearn
from sklearn.datasets import make_blobs

def generate_data(n_samples, flagc):
    if flagc == 1:
        random_state = 365
        X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
    elif flagc == 2:
        random_state = 148
        X, y = make_blobs(n_samples=n_samples, random_state=random_state)
        transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
        X = np.dot(X, transformation)
    elif flagc == 3:
        random_state = 148
        X, y = make_blobs(n_samples=n_samples, centers=4, cluster_std=[1.0, 2.5, 0.5, 3.0], random_state=random_state)
    elif flagc == 4:
        X, y = datasets.make_circles(n_samples=n_samples, factor=.5, noise=.05)
    elif flagc == 5:
        X, y = datasets.make_moons(n_samples=n_samples, noise=.05)
    else:
        X = []
    return X, y

X, y = generate_data(500, flagc=5)

plt.scatter(X[:, 0], X[:, 1], s=10)
plt.title("Generirani podaci")
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()

kmeans = KMeans(n_clusters=3, random_state=42)
y_kmeans = kmeans.fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=10, cmap='viridis')

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='cyan', s=200, alpha=0.75, marker='X')
plt.title("Clusteri")
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()

#Promjenom načina generiranja koda dobivamo drugačije oblike koje se pomoću kmeans ne grupiraju idealno