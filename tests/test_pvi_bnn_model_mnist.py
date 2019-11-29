import os
import sys
import torch
import logging
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets, transforms

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# add formatter to ch
ch.setFormatter(formatter)

# add ch to logger
logger.addHandler(ch)

sys.path.append(os.getcwd())

from data.distributed_mnist import DistributedMNIST
from src.methods.pvi.model_bnn import BNNModel
import src.methods.exp_family.diagonal_gaussian as dg
import src.methods.exp_family.utils as utils
from src.methods.pvi.client import PVIClient
from src.methods.pvi.server import SyncronousPVIParameterServer

torch.manual_seed(0)

data_root = '' 

num_clients = 10

model_hyperparameters = {
            'in_features': 28*28,
            'hidden_sizes': [200],
            'out_features': 10,
            'train_samples': 10,
            'test_samples': 10,
            'batch_size': 200,
            'type': 'classification',
            'lr': 5e-3,
            'N_sync': 100,
            'device': 'cpu'
        }


train_datas = [ DistributedMNIST(
    '/data/hemispingus/mhutchin/Projects/distributed-bayesian-learning/data',
    shard=i,
    num_shards=num_clients,
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
for i in range(num_clients)
]


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

print(train_datas[0])
print(test_data)


base_model = BNNModel(
    parameters=None, 
    hyperparameters=model_hyperparameters
)

parameters = base_model.get_parameters()
# t_i = utils.list_const_div(num_clients, t_i)
prior = dg.list_uniform_prior(parameters, 0., 1.)

global_t_i = utils.list_sub(parameters, prior)
client_t_i = utils.list_const_div(num_clients, global_t_i)

clients = [PVIClient(
    BNNModel,
    train_datas[i],
    t_i=client_t_i,
    model_parameters=None,
    model_hyperparameters=model_hyperparameters,
    hyperparameters={
        'damping_factor': 0.95
    },
    metadata=None
)
for i in range(num_clients)
]

parameters = utils.list_add(*[client.t_i for client in clients], prior)

server = SyncronousPVIParameterServer(
    BNNModel,
    prior,
    clients,
    model_parameters=parameters,
    model_hyperparameters=model_hyperparameters,
    hyperparameters={
        'damping_factor': 0.95,
        'damping_decay': 1.718e-3
    }
)

for epoch in range(5):
    server.tick()
    
    test_log_loss, test_reg_loss = server.evaluate(
        test_data
    )
    logger.info(f'Epoch {epoch}: \t log loss {test_log_loss: 0.6f}, accuracy {test_reg_loss*100:0.3f}, damping {server.current_damping_factor}')

results_dir = 'results/test'
os.makedirs(results_dir, exist_ok=True)

server.model.save_model(results_dir)

new_model = BNNModel.load_model(results_dir)

test_log_loss, test_reg_loss = new_model.evaluate(
    test_data
)
logger.info(f'Epoch {epoch}: \t log loss {test_log_loss: 0.6f}, accuracy {test_reg_loss*100:0.3f}, damping {server.current_damping_factor}')