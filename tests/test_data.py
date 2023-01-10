import pytest
from tests import _PATH_DATA
import torch
import os

@pytest.mark.skipif(not os.path.exists(_PATH_DATA + '/processed/trainset.pt'), reason="Data files not found")
@pytest.mark.skipif(not os.path.exists(_PATH_DATA + '/processed/testset.pt'), reason="Data files not found")
def test_data():
    trainset = torch.load(_PATH_DATA + '/processed/trainset.pt')
    testset = torch.load(_PATH_DATA + '/processed/testset.pt')
    assert len(trainset) == 25000, "Trainset have incorrect length"
    assert len(testset) == 5000, "Testst have incorrect length"
    for data_point, _ in trainset:
        assert data_point.shape == (28,28)

    assert len(trainset.tensors[1].unique()) == 10, "All classes are not represented"
