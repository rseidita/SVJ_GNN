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

epochs = 100

num_features = 12 

scale = 0.55

damping = 5

dropout=0.04

from architectures.GraphAE import graphAE

model=graphAE(
    num_features=num_features,
    architecture=(100,100,100,10),
    dropout=dropout,
    batch_norm=True
)
import torch


'''loss_features=torch.nn.MSELoss()

loss_edges=

loss_function= loss_edges + loss_features'''

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set number of threads
torch.set_num_threads(12)

from utils.ConstraintManager import MDMMConstraint, ConstraintsManager

learning_rate = 5e-3

use_mdmm = True

constraint_variables = [
    "nodes",
    "edges"
]

nodes_constraint=MDMMConstraint.create_constraint(
    constraint_variables[0],
    scale,
    damping,
    model,
    )
edges_constraint=MDMMConstraint.create_constraint(
    constraint_variables[1],
    scale,
    damping,
    model,
)


constraints_manager = ConstraintsManager(constraint_variables)
constraints_manager.add(nodes_constraint)
constraints_manager.add(edges_constraint)
