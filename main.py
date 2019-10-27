# visualize kmeans algorithm
from celluloid import Camera  # useful tool to visualize matplots
from sklearn.datasets.samples_generator import make_blobs
from matplotlib import pyplot as plt
import numpy as np


def create_data(type="sklearn"):
    if type == "sklearn":
        X, y = make_blobs(n_samples=500, centers=4, cluster_std=0.95)
        # plt.scatter(X[:, 0], X[:, 1])
        # plt.draw()
        # plt.pause(0.001)
        # input("Press enter to continue: ")
        return X


def run_kmeans(X, num_clusters=4):
    centroids = np.random.uniform(low=np.min(X), high=np.max(X), size=(num_clusters, 2))
    plt.scatter(X[:, 0], X[:, 1])
    plt.scatter(centroids[:, 0], centroids[:, 1], color="red")
    plt.show()


def main():
    X = create_data()  # numpy array of 2d numpy arrays
    run_kmeans(X)


if __name__ == "__main__":
    main()
