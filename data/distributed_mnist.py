from torchvision import datasets

from src.distribution.homogenous_split import make_split_indicies

class DistributedMNIST(datasets.MNIST):
    """Makes the MNIST dataset into a series of shards for 
    comparing distributed learning.
    
    Arguments:
        datasets {[type]} -- [description]
    """

    def __init__(
        self, 
        root, 
        shard,
        num_shards,
        num_validation=0,
        seed=0,
        train=True, transform=None, target_transform=None, download=True):
        super(DistributedMNIST, self).__init__(root, train, transform, target_transform, download)

        if train:
            indicies = make_split_indicies(
                self.data.shape[0],
                shard,
                num_shards, 
                seed,
                num_validation,
                0
            )
            self.data = self.data[indicies]
            self.targets = self.targets[indicies]