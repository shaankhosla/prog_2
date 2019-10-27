# visualize kmeans algorithm
from celluloid import Camera  # useful tool to visualize matplots
from sklearn.datasets.samples_generator import make_blobs
from matplotlib import pyplot as plt
import numpy as np


def create_data(type="sklearn"):
    if type == "sklearn":
        X, y = make_blobs(n_samples=100, centers=4, cluster_std=0.60, random_state=0)
        plt.scatter(X[:, 0], X[:, 1])
        plt.draw()
        plt.pause(0.001)
        input("Press [enter] to continue.")
        return X


def run_kmeans(X, num_clusters=3):
    centroids = np.random.uniform(low=np.min(X), high=np.max(X), size=(num_clusters, 2))
    pass


def main():
    X = create_data()  # numpy array of 2d numpy arrays
    run_kmeans(X)


if __name__ == "__main__":
    main()
