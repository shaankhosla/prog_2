import numpy as np
import matplotlib.pyplot as plt

ITERATION_COUNT = 4
SAMPLE_COUNT = 100
MEAN_COUNT = 3

data = np.zeros((SAMPLE_COUNT, 2))
means = np.random.randn(MEAN_COUNT, 2)
trueMeans = np.random.randn(MEAN_COUNT, 2) * 5.0

trueCovs = np.random.randn(MEAN_COUNT, 2, 2)
#Make sure covs are symmetric positive definite
for i in range(MEAN_COUNT):
    trueCovs[i] = trueCovs[i] * trueCovs[i].transpose()
    trueCovs[i] = trueCovs[i] + 2 * np.eye(2)

assigns = np.zeros((SAMPLE_COUNT, MEAN_COUNT))
trueAssigns = np.zeros((SAMPLE_COUNT))

for i in range(SAMPLE_COUNT):
    randomCluster = np.random.randint(MEAN_COUNT)
    data[i] = np.random.multivariate_normal(trueMeans[randomCluster], trueCovs[randomCluster], 1)
    trueAssigns[i] = randomCluster

for i in range(ITERATION_COUNT):
    dists = np.sum((data[:, :, np.newaxis] - means.transpose()) ** 2.0, axis = 1)
    assigns = np.argsort(dists, axis = 1)[:, 0]

    first = data[np.where(assigns == 0)]
    second = data[np.where(assigns == 1)]
    third = data[np.where(assigns == 2)]

    plt.plot(first[:, 0], first[:, 1], 'ro')
    plt.plot(second[:, 0], second[:, 1], 'go')
    plt.plot(third[:, 0], third[:, 1], 'bo')

    plt.plot(means[0, 0], means[0, 1], 'rx', markersize = 18)
    plt.plot(means[1, 0], means[1, 1], 'gx', markersize = 18)
    plt.plot(means[2, 0], means[2, 1], 'bx', markersize = 18)

    plt.pause(1.0)

    for j in range(MEAN_COUNT):
        means[j, :] = np.mean(data[np.where(assigns == j)], axis = 0)

    plt.clf()

    plt.plot(first[:, 0], first[:, 1], 'ro')
    plt.plot(second[:, 0], second[:, 1], 'go')
    plt.plot(third[:, 0], third[:, 1], 'bo')

    plt.plot(means[0, 0], means[0, 1], 'rx', markersize = 18)
    plt.plot(means[1, 0], means[1, 1], 'gx', markersize = 18)
    plt.plot(means[2, 0], means[2, 1], 'bx', markersize = 18)

    plt.pause(1.0)

    if i != ITERATION_COUNT - 1:
        plt.clf()

plt.figure()

first = data[np.where(trueAssigns == 0)]
second = data[np.where(trueAssigns == 1)]
third = data[np.where(trueAssigns == 2)]

plt.plot(first[:, 0], first[:, 1], 'ro')
plt.plot(second[:, 0], second[:, 1], 'go')
plt.plot(third[:, 0], third[:, 1], 'bo')

plt.plot(trueMeans[0, 0], trueMeans[0, 1], 'rx', markersize = 18)
plt.plot(trueMeans[1, 0], trueMeans[1, 1], 'gx', markersize = 18)
plt.plot(trueMeans[2, 0], trueMeans[2, 1], 'bx', markersize = 18)

'''
plt.plot(means[0, 0], means[0, 1], 'rx')
plt.plot(means[1, 0], means[1, 1], 'gx')
plt.plot(means[2, 0], means[2, 1], 'bx')

first = data[np.where(assigns == 0)]
second = data[np.where(assigns == 1)]
third = data[np.where(assigns == 2)]

plt.plot(first[:, 0], first[:, 1], 'ro')
plt.plot(second[:, 0], second[:, 1], 'go')
plt.plot(third[:, 0], third[:, 1], 'bo')
'''

plt.show()

