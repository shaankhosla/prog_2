# visualize kmeans algorithm
from celluloid import Camera  # useful tool to visualize matplots
from sklearn.datasets.samples_generator import make_blobs
from matplotlib import pyplot as plt
import numpy as np
import matplotlib as mp
from numba import jit

COLORS = ["grey", "blue", "green", "purple", "pink", "yellow", "black", "magenta"]


def create_data(centers, std):
    X, y = make_blobs(n_samples=2500, centers=centers, cluster_std=std)
    # plt.scatter(X[:, 0], X[:, 1])
    # plt.show()
    return X


@jit(nopython=True)
def distance(x, y):
    return np.linalg.norm(x - y)


def find_clusters(X, centroids):
    index = []
    for item in X:
        minimum = 999999999999
        for centroid_ct in range(len(centroids)):
            d = distance(item, centroids[centroid_ct])
            if d < minimum:
                closest_centroid = centroid_ct
                minimum = d
        index.append(closest_centroid)
    return index


def update_centroids(X, index, num_centroids):
    centroids = []
    for i in range(num_centroids):
        cluster = [X[j] for j in range(len(index)) if index[j] == i]
        if cluster != []:
            centroids.append(np.average(cluster, axis=0))
        else:
            print('Centroid had no points nearest to it, choosing random value to use instead..')
            centroids.append(X[np.random.choice(len(X))])

    return np.asarray(centroids)


def run_kmeans(X, num_centroids, centers, std):
    centroids = np.random.uniform(low=np.min(X), high=np.max(X), size=(num_centroids, 2))
    index = find_clusters(X, centroids)
    (ct, camera) = (0, Camera(plt.figure()))

    for i in range(3):  # slows down the mp4 video
        plt.scatter(X[:, 0], X[:, 1], c=index, cmap=mp.colors.ListedColormap(COLORS))
        plt.scatter(centroids[:, 0], centroids[:, 1], color="r")
        camera.snap()

    while(True):
        ct += 1
        old_centroids = centroids
        centroids = update_centroids(X, index, num_centroids)
        if (np.array_equal(centroids, old_centroids)) or (ct == 150):
            break
        index = find_clusters(X, centroids)

        for i in range(2):  # slows down the mp4 video
            plt.scatter(X[:, 0], X[:, 1], c=index, cmap=mp.colors.ListedColormap(COLORS))
            plt.scatter(centroids[:, 0], centroids[:, 1], color="r")
            camera.snap()

    anim = camera.animate(blit=True)
    anim.save("./output/iter" + str(ct) + '-centroids' + str(num_centroids) + '-centers' + str(centers) + '-std' + str(std) + ".mp4")
    plt.close('all')
    return ct


def main():
    for centers in [2, 4, 8]:
        for std in [.5, 1.5, 2, 2.25, 2.75]:
            X = create_data(centers, std)
            for num_centroids in [3, 5, 8]:
                run_kmeans(X, num_centroids, centers, std)


if __name__ == "__main__":
    main()
    print('\a')
