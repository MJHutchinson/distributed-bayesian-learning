import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets, transforms

sys.path.append(os.getcwd())

from data.distributed_mnist import DistributedMNIST
from src.models.bnn.model import BayesianMLPReparam


def train_epoch(model, device, num_samples, num_train, prior_mean, prior_std, train_loader, optimiser, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        num_batch, _ = data.shape
        optimiser.zero_grad()

        output = model(data, num_samples=num_samples)

        target = target.unsqueeze(1).repeat_interleave(num_samples, dim=1)

        nll = F.cross_entropy(output.view(-1, target.shape[-1]), target.view(-1), reduction='mean')
        kl = model.kl() 
        loss = nll + (kl / num_train)

        loss.backward()
        optimiser.step()

        if (batch_idx+1) % 100 == 0:
            print(f'Train epoch: {epoch}, batch {batch_idx} of {len(train_loader)} \t loss: {loss:0.6f} \t nll: {nll:0.6f} \t KL: {kl:0.6f}') 


def test_epoch(model, device, num_samples, num_test, prior_mean, prior_std, test_loader, epoch):
    model.eval()
    test_nll = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            num_batch, _ = data.shape

            output = model(data, num_samples=num_samples)
            target_rep = target.unsqueeze(1).repeat_interleave(num_samples, dim=1)

            nll = F.cross_entropy(output.view(-1, output.shape[-1]), target_rep.view(-1), reduction='sum')
            test_nll += nll

            pred = output.mean(dim=1).argmax(dim=1)
            correct += pred.eq(target).sum().item()

        kl = model.kl()
        
    print(f'Test epoch: {epoch} avg nll: {(test_nll / num_test):0.6f}, kl: {kl:0.6f}, acc: {(correct /  num_test * 100):0.4f}')


data_root = '' 

train_data = DistributedMNIST(
    '/data/hemispingus/mhutchin/Projects/distributed-bayesian-learning/data',
    shard=0,
    num_shards=1,
    num_validation=0,
    seed=0,
    download=True,
    train=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: x.reshape(-1))
    ])
)


test_data = DistributedMNIST(
    '/data/hemispingus/mhutchin/Projects/distributed-bayesian-learning/data',
    shard=0,
    num_shards=1,
    num_validation=0,
    seed=0,
    download=True,
    train=False,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: x.reshape(-1))
    ])
)

print(train_data)
print(test_data)

train_dataloader = DataLoader(train_data, 200, shuffle=True, num_workers=4)
test_dataloader = DataLoader(test_data, 200, shuffle=False, num_workers=4)

model = BayesianMLPReparam(28*28, [1024,1024,1024], 10, init_logvar=-11., init_mean_variance=0.1)

optimiser = torch.optim.Adam(model.parameters() ,lr=1e-3)

for epoch in range(100):
    train_epoch(model, 'cpu', 10, len(train_data), 0., 1., train_dataloader, optimiser, epoch)
    test_epoch(model, 'cpu', 11, len(test_data), 0., 1., test_dataloader, epoch)