# visualize kmeans algorithm
from celluloid import Camera  # useful tool to visualize matplots
from sklearn.datasets.samples_generator import make_blobs
from matplotlib import pyplot as plt
import numpy as np
import sys


def create_data(type="sklearn"):
    if type == "sklearn":
        X, y = make_blobs(n_samples=500, centers=4, cluster_std=0.95)
        # plt.scatter(X[:, 0], X[:, 1])
        # plt.draw()
        # plt.pause(0.001)
        # input("Press enter to continue: ")
        return X


def find_clusters(X, centroids):
    index = []
    for item in X:
        minimum = 999999999999
        for centroid_ct in range(len(centroids)):
            distance = np.linalg.norm(item - centroids[centroid_ct])
            if distance < minimum:
                closest_centroid = centroid_ct
                minimum = distance
        index.append(closest_centroid)
    return index


def update_centroids(X, index, num_clusters):
    centroids = []
    for i in range(num_clusters):
        cluster = [X[j] for j in range(len(index)) if index[j] == i]
        centroids.append(np.average(cluster, axis=0))
    return np.asarray(centroids)


def run_kmeans(X, num_clusters=4):
    centroids = np.random.uniform(low=np.min(X), high=np.max(X), size=(num_clusters, 2))
    index = find_clusters(X, centroids)
    camera = Camera(plt.figure())
    ct = 0
    while(True):
        ct += 1
        centroids = update_centroids(X, index, num_clusters)
        index = find_clusters(X, centroids)
        plt.scatter(X[:, 0], X[:, 1])
        plt.scatter(centroids[:, 0], centroids[:, 1], color="red")
        camera.snap()
        if ct == 100:
            break
    anim = camera.animate(blit=True)


def main():
    X = create_data()  # numpy array of 2d numpy arrays
    run_kmeans(X)


if __name__ == "__main__":
    main()
