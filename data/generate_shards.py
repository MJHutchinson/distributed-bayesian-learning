import os
import sys
sys.path.append(os.getcwd())

from torchvision import datasets, transforms

def genereate_shards(data_config, seed):
    if data_config['dataset'] == 'MNIST':
        from data.distributed_mnist import DistributedMNIST

        num_shards = data_config['num_shards']
        data_root = data_config['data_root']
        num_validation = data_config['num_validation']

        train_shards = [
            DistributedMNIST(
                root=data_root,
                shard=shard,
                num_shards=num_shards,
                num_validation=num_validation,
                seed=seed,
                train=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                    transforms.Lambda(lambda x: x.reshape(-1))
                ]),
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
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                    transforms.Lambda(lambda x: x.reshape(-1))
                ]),
                download=True
            )

        test_shard = DistributedMNIST(
                root=data_root,
                shard='validation',
                num_shards=num_shards,
                num_validation=num_validation,
                seed=seed,
                train=False,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                    transforms.Lambda(lambda x: x.reshape(-1))
                ]),
                download=True
            )

        return train_shards, validation_shard, test_shard