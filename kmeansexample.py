import numpy as np
import matplotlib.pyplot as plt

ITERATION_COUNT = 4
SAMPLE_COUNT = 100
MEAN_COUNT = 3

data = np.random.randn(SAMPLE_COUNT, 2)
means = np.random.randn(MEAN_COUNT, 2)
assigns = np.zeros((SAMPLE_COUNT, MEAN_COUNT))

#plt.figure()
#plt.show()

for i in range(ITERATION_COUNT):
    dists = np.sum((data[:, :, np.newaxis] - means.transpose()) ** 2.0, axis = 1)
    assigns = np.argsort(dists, axis = 1)[:, 0]

    first = data[np.where(assigns == 0)]
    second = data[np.where(assigns == 1)]
    third = data[np.where(assigns == 2)]

    plt.plot(first[:, 0], first[:, 1], 'ro')
    plt.plot(second[:, 0], second[:, 1], 'go')
    plt.plot(third[:, 0], third[:, 1], 'bo')

    plt.pause(1.0)

    for j in range(MEAN_COUNT):
        means[j, :] = np.mean(data[np.where(assigns == j)], axis = 0)

    plt.plot(means[0, 0], means[0, 1], 'rx')
    plt.plot(means[1, 0], means[1, 1], 'gx')
    plt.plot(means[2, 0], means[2, 1], 'bx')

    plt.pause(1.0)

    if i != ITERATION_COUNT - 1:
        plt.clf()

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
