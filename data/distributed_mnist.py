from torchvision import datasets

from src.distribution.homogenous_split import make_split_indicies

class DistributedMNIST(datasets.MNIST):
    """Makes the MNIST dataset into a series of shards for 
    comparing distributed learning. Uses a speed up as per
    https://gist.github.com/y0ast/f69966e308e549f013a92dc66debeeb4
    and does not allow transforms to be used.
    """

    def __init__(
        self, 
        root, 
        shard,
        num_shards,
        num_validation=0,
        seed=0,
        flatten=True,
        device=None,
        train=True, 
        download=True):
        super(DistributedMNIST, self).__init__(root, train=train, download=download)

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

        # Scale data to [0,1]
        self.data = self.data.float().div(255)
        
        # Normalize it with the usual MNIST mean and std
        self.data = self.data.sub_(0.1307).div_(0.3081)

        # Flatten the data if required. If not, assume we're
        # using a CNN and want a channels dimension
        if flatten:
            num_points, _,_ = self.data.shape
            if num_points != 0:
                self.data = self.data.view(num_points, -1)
        else:
            self.data.unsqueeze(1)
        
        # Put both data and targets on GPU in advance
        if device is not None:
            self.data, self.targets = self.data.to(device), self.targets.to(device)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        return img, target