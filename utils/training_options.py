###########################################
###########################################
# This config file holds all necessary
# info for training the GraphAutoEncoder model(s)
###########################################
###########################################

# Define the subset of variables to train on
# All of these are also required to be in the
# overall list of features in data_loading_options.py


batch_size=128

epochs = 1

num_features = 12 

epsilon=0.55

scale=30

damping=5

from architectures.GraphAE import graphAEBatchNorm

model=graphAEBatchNorm(
    num_features=num_features,
    architecture=(100,100,100,10),
    dropout=0.04,
    batch_norm=True,
)
import torch


'''loss_features=torch.nn.MSELoss()

loss_edges=

loss_function= loss_edges + loss_features'''

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set number of threads
torch.set_num_threads(12)

learning_rate = 5e-3

use_mdmm = True

constraint_variables = [
    "nodes",
    #"edges"
]
import mdmm

nodes_constraint=mdmm.MaxConstraint(
    lambda x, edges: model.features_loss(x, edges),
    max=epsilon,
    scale=scale,
    damping=damping,
    )
from torch_geometric.utils import batched_negative_sampling

edge_constraint = mdmm.MaxConstraint(
    lambda x, edges, batch: model.edges_loss(x, edges, torch.tensor(batched_negative_sampling(edges, batch), dtype=torch.int64)),
    max=epsilon,
    scale=scale,
    damping=damping,
    )
