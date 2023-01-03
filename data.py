import torch
import numpy as np


def mnist():
    # exchange with the corrupted mnist dataset
    xtrain = np.load('/Users/benj3542/dtu_mlops/data/corruptmnist/train_0.npz')['images']
    ytrain = np.load('/Users/benj3542/dtu_mlops/data/corruptmnist/train_0.npz')['labels']

    for i in range(1,5): 
        xtrain = np.concatenate((xtrain, np.load(f'/Users/benj3542/dtu_mlops/data/corruptmnist/train_{i}.npz')['images']), axis = 0)
        ytrain = np.concatenate((ytrain, np.load(f'/Users/benj3542/dtu_mlops/data/corruptmnist/train_{i}.npz')['labels']), axis = 0)
    xtest = np.load('/Users/benj3542/dtu_mlops/data/corruptmnist/test.npz')['images']
    ytest = np.load('/Users/benj3542/dtu_mlops/data/corruptmnist/test.npz')['labels']

    trainset = torch.utils.data.TensorDataset(torch.from_numpy(xtrain).float(), torch.from_numpy(ytrain).long())
    testset = torch.utils.data.TensorDataset(torch.from_numpy(xtest).float(), torch.from_numpy(ytest).long())

    return trainset, testset
    
if __name__ == "__main__":
    mnist()
