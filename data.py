import numpy as np

def generateData(numOfSamples,  numOfDims, numOfClusters):
    data = np.zeros((numOfSamples, numOfDims))
    trueAssigns = np.zeros((numOfSamples))
    
    #5.0 is an arbitrary spacing for the values
    trueMeans = np.random.randn(numOfClusters, numOfDims) * 5.0
    trueCovs = np.random.randn(numOfClusters, numOfDims, numOfDims)

    #Make sure covs are symmetric positive definite
    for i in range(numOfClusters):
        trueCovs[i] = trueCovs[i] * trueCovs[i].transpose()
        trueCovs[i] = trueCovs[i] + numOfDims * np.eye(numOfDims)

    for i in range(numOfSamples):
        randomCluster = np.random.randint(numOfClusters)
        data[i] = np.random.multivariate_normal(trueMeans[randomCluster], trueCovs[randomCluster], 1)
        trueAssigns[i] = randomCluster

    return data, trueAssigns
