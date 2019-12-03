import os
import sys
sys.path.append(os.getcwd())

from torchvision import datasets, transforms

def genereate_shards(data_config, seed, device=None):
    if data_config['dataset'] == 'MNIST':
        from data.distributed_mnist import DistributedMNIST

        num_shards = data_config['num_shards']
        data_root = data_config['data_root']
        num_validation = data_config['num_validation']
        flatten = data_config['flatten']

        train_shards = [
            DistributedMNIST(
                root=data_root,
                shard=shard,
                num_shards=num_shards,
                num_validation=num_validation,
                seed=seed,
                train=True,
                flatten=flatten,
                device=device,
                download=True
            )
            for shard in range(num_shards)
        ]

        validation_shard = DistributedMNIST(
                root=data_root,
                shard='validation',
                num_shards=num_shards,
                num_validation=num_validation,
                seed=seed,
                train=True,
                flatten=flatten,
                device=device,
                download=True
            )

        test_shard = DistributedMNIST(
                root=data_root,
                shard='validation',
                num_shards=num_shards,
                num_validation=num_validation,
                seed=seed,
                train=False,
                flatten=flatten,
                device=device,
                download=True
            )

        return train_shards, validation_shard, test_shard