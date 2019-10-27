# visualize kmeans algorithm
from celluloid import Camera  # useful tool to visualize matplots
from sklearn.datasets.samples_generator import make_blobs
from matplotlib import pyplot as plt


def create_data(type="sklearn"):

    if type == "sklearn":
        X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
        plt.scatter(X[:, 0], X[:, 1])
        plt.show()
        print(y)


def main():
    create_data()


if __name__ == "__main__":
    main()
