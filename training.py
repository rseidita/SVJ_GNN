from comet_ml import Experiment

experiment = Experiment(
  api_key="bNAc34AOCkkuz1iKalBL4nISK",
  project_name="GNN_training",
  workspace="celesteholm"
)

tag1 = "GINConv"
tag2 = "lund_vars"
tag3 = "lund_vars_and_Efr"
tag4 = "gpu_eff_test"
tag5 = "mdmm"
tag6 = "disk_space_fix"
tag7 = "edge_loss_only"
tag8 = "constrain_egde_loss"
tag9 = "offline_test"
tag10 = "with_4momenta"
tag11 = "node_loss_only"
tag12 = "directed"
tag13 = "large"
all_tags = [tag7, tag10, tag13]
experiment.add_tags(all_tags)

qcd_path = "/pnfs/psi.ch/cms/trivcat/store/user/ceholm/forTraining_new/QCD_new/"
signal_base_path = "/pnfs/psi.ch/cms/trivcat/store/user/ceholm/forTraining_new/SVJ_new/"
ttjets_path = "/pnfs/psi.ch/cms/trivcat/store/user/ceholm/forTraining_new/TTJets_new/"


signals = {
    "SVJ_700_0p3" : signal_base_path + "SVJ_mMed-700GeV_mDark-20GeV_rinv-0.3_alpha-peak/",
    "SVJ_700_0p5" : signal_base_path + "SVJ_mMed-700GeV_mDark-20GeV_rinv-0.5_alpha-peak/",
    "SVJ_700_0p7" : signal_base_path + "SVJ_mMed-700GeV_mDark-20GeV_rinv-0.7_alpha-peak/",

    "SVJ_800_0p3" : signal_base_path + "SVJ_mMed-800GeV_mDark-20GeV_rinv-0.3_alpha-peak/",
    "SVJ_800_0p5" : signal_base_path + "SVJ_mMed-800GeV_mDark-20GeV_rinv-0.5_alpha-peak/",
    "SVJ_800_0p7" : signal_base_path + "SVJ_mMed-800GeV_mDark-20GeV_rinv-0.7_alpha-peak/",

    "SVJ_900_0p3" : signal_base_path + "SVJ_mMed-900GeV_mDark-20GeV_rinv-0.3_alpha-peak/",
    "SVJ_900_0p5" : signal_base_path + "SVJ_mMed-900GeV_mDark-20GeV_rinv-0.5_alpha-peak/",
    "SVJ_900_0p7" : signal_base_path + "SVJ_mMed-900GeV_mDark-20GeV_rinv-0.7_alpha-peak/",

    "SVJ_1000_0p3" : signal_base_path + "SVJ_mMed-1000GeV_mDark-20GeV_rinv-0.3_alpha-peak/",
    "SVJ_1000_0p5" : signal_base_path + "SVJ_mMed-1000GeV_mDark-20GeV_rinv-0.5_alpha-peak/",
    "SVJ_1000_0p7" : signal_base_path + "SVJ_mMed-1000GeV_mDark-20GeV_rinv-0.7_alpha-peak/",

    "SVJ_1500_0p3" : signal_base_path + "SVJ_mMed-1500GeV_mDark-20GeV_rinv-0.3_alpha-peak/",
    "SVJ_1500_0p5" : signal_base_path + "SVJ_mMed-1500GeV_mDark-20GeV_rinv-0.5_alpha-peak/",
    "SVJ_1500_0p7" : signal_base_path + "SVJ_mMed-1500GeV_mDark-20GeV_rinv-0.7_alpha-peak/",

    "SVJ_3000_0p3" : signal_base_path + "SVJ_mMed-3000GeV_mDark-20GeV_rinv-0.3_alpha-peak/",
    "SVJ_3000_0p5" : signal_base_path + "SVJ_mMed-3000GeV_mDark-20GeV_rinv-0.5_alpha-peak/",
    "SVJ_3000_0p7" : signal_base_path + "SVJ_mMed-3000GeV_mDark-20GeV_rinv-0.7_alpha-peak/",
}


from utils.JetGraphProducer import JetGraphProducer
from utils.LundTreeUtilities import OnTheFlyNormalizer
import torch

# print(f"CUDA is available: {torch.cuda.is_available()}")
# Train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import argparse
import json

parser = argparse.ArgumentParser()
# parser.add_argument(
#     "--alpha_train",
#     default=150,
#     help="Relative weight of nodes loss in training",
#     type=float,
#     )
# parser.add_argument(
#     "--alpha_test",
#     default=150,
#     help="Relative weight of nodes loss in testing",
#     type=float,
#     )


parser.add_argument(
    "--learning_rate",
    default=1e-3,
    help="Learning rate to be passed to optimizer",
    type=float,
    )
parser.add_argument(
    "--batch_size",
    default=64,
    help="Training and validation batch size",
    type=int,
    )
parser.add_argument(
    "--features_loss_target",
    default=0.01,
    help="Training and validation batch size",
    type=float,
    )
parser.add_argument(
    "--architecture",
    default="[256, 128, 64, 32, 16, 8, 4]",
    help="Graph autoencoder architecture",
    type=str,
    )

parser.add_argument(
    "--autoencoder",
    default="architectures/GraphAE.py",
    help="Graph autoencoder to use (GCN or GIN)",
    type=str,
    )

args = parser.parse_args()

# alpha_test_val= args.alpha_test
# alpha_train_val = args.alpha_train

if args.autoencoder=="architectures/GraphAE.py":
    tag14 = "GCNConv"
elif args.autoencoder =="architectures/GraphAE_GIN.py":
    tag14 = "GINConv"

experiment.add_tags([tag14])

import ast

LR = args.learning_rate
batch_size = args.batch_size

args.architecture = ast.literal_eval(args.architecture)
architecture = args.architecture

features_loss_target = args.features_loss_target

hyperparams = {
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "features_loss_target": args.features_loss_target,
        "architecture": args.architecture,
    }

experiment.log_parameters(hyperparams)


arch_string = str(architecture[0]) + "_to_" + str(architecture[-1])

# Produce the graphs. The call to JetGraphProducer will produce the graphs and
# save them in the specified directory if it is not already present.
qcd = JetGraphProducer(
        qcd_path, # Path to the directory containing the root files to be processed
        "QCD_large", # Name to be given to the produced dataset (or the name of the dataset if it is already present)
        n_store_jets=2, # Number of jets to process per event, in order of decreasing pT
        # max_events_to_process=100, # Maximum number of events to process (set to None to process all events)
        save_event_number=True,
        use_lund_decomp=True, # Whether to use the Lund decomposition
        n_lund_vars=5, # Number of variables to store in the Lund decomposition (3 or 5)
        save_energy_fractions=True, # Whether to save the energy fractions of the pseudojets in the Lund graph
        save_4vectors_in_lund_tree=True, # Whether to save the 4-vectors of the pseudojets in the Lund graph
        weights="xsec", # Weights to be applied to the graphs (None, "xsec")
        extra_obs_to_save_per_event=["MT01FatJetMET"],
        verbose=True, # Whether to print the progress of the processing
        label=0, # Label to be assigned to the graphs (for supervised learning)
        directed=False,
    )

# ttjets = JetGraphProducer(
#         ttjets_path, 
#         "TTJets_lund_vars", 
#         n_store_jets=2, 
#         save_event_number=True,
#         # max_events_to_process=100, 
#         use_lund_decomp=True, 
#         n_lund_vars=5, 
#         save_energy_fractions=False, 
#         save_4vectors_in_lund_tree=False, 
#         weights="xsec", 
#         extra_obs_to_save_per_event=["MT01FatJetMET"],
#         verbose=True, 
#         label=0, 
#     )

signal_graphs = {}
signal_name = "SVJ_1000_0p5"
signal_path = signals["SVJ_1000_0p5"]
# for signal_name, signal_path in signals.items():
signal_graphs[signal_name] = (JetGraphProducer(
    signal_path,
    f"{signal_name}_large",
    n_store_jets=2,
    save_event_number=True,
    # max_events_to_process=100,
    use_lund_decomp=True,
    n_lund_vars=5,
    save_energy_fractions=True,
    save_4vectors_in_lund_tree=True,
    weights="xsec",
    extra_obs_to_save_per_event=["MT01FatJetMET"],
    verbose=True,
    label=1,
    directed=False,
))


# Compute the means and stds for the normalizer (if required)

print("Computing normalization...")

means, stds = 0., 0.
weights = []

for graph in qcd:
    means += graph.x.sum(dim=0)*graph.w
    weights.append(graph.w)
    
# for graph in ttjets:
#     means += graph.x.sum(dim=0)*graph.w
#     weights.append(graph.w)


import numpy as np

num_nodes_qcd = np.array([graph.num_nodes for graph in qcd])
# num_nodes_ttjets = np.array([graph.num_nodes for graph in ttjets])
# num_nodes = np.concatenate((num_nodes_qcd, num_nodes_ttjets))

sum_weights = np.repeat(weights, num_nodes_qcd).sum()

means /= sum_weights

for graph in qcd:
    stds += ((graph.x - means)**2).sum(dim=0)*graph.w

# for graph in ttjets:
#     stds += ((graph.x - means)**2).sum(dim=0)*graph.w

stds /= sum_weights
stds = torch.sqrt(stds)

normalizer = OnTheFlyNormalizer(["x"], means, stds, device=device)

import pickle
normalizer_dict = {
    "means": means,
    "stds": stds,
}
with open(f"models/10_07_bs_{batch_size}_lr_{LR}_flt_{features_loss_target}_arch_{arch_string}_{tag14}_{tag7}_{tag10}_{tag13}.pkl", "wb") as outFile:
    pickle.dump(normalizer_dict, outFile)




from data_loading.graphDataLoader import GraphDataLoader

# Create the data loaders for training and testing. Note how
# you can concatenate multiple datasets in a single loader and how
# you can specify the window of each dataset to load.

print("Preparing training dataloader...")

loader_training = GraphDataLoader(
    [
        [
            qcd_path, # path to the dataset
            "QCD_large", # name of the dataset (you can process the same raw data with multiple settings given each a different name)
        ],
        # [
        #     ttjets_path, # path to the dataset
        #     "TTJets_large", # name of the dataset (you can process the same raw data with multiple settings given each a different name)
        # ],
    ],
    window_to_load=[[0., 0.6], ],    #[0., 0.6]],
    batch_size=batch_size,
    #dataset_fraction=[0.99, 0.01,],
    normalizer=normalizer,
    device=device,
    shuffle=True,
)


permutation = loader_training.get_permutation()

print("Preparing validation dataloader...")

loader_validation = GraphDataLoader(
    [
        [qcd_path, "QCD_large"],
        # [ttjets_path, "TTJets_large"],
    ],
    
    window_to_load=[[0.6, 0.7], ],  #  [0.6, 0.7],],
    #batch_size=batch_size,
    #dataset_fraction=[0.99, 0.01,],
    normalizer=normalizer,
    device=device,
    permutation=permutation,
)

# print("Preparing testing dataloader...")

# loader_testing = GraphDataLoader(
#     [
#         [qcd_path, "QCD_large"],
#         # [ttjets_path, "TTJets_large"],
#     ],
    
#     window_to_load=[[0.7, 1.],],      #[0.7, 1.],],
#     #dataset_fraction=[0.99, 0.01,],
#     normalizer=normalizer,
#     device=device,
#     permutation=permutation,
# )

print("Saving permutation...")

permutation_dict = {"permutation": permutation,}
with open(f"models/perm_10_07_bs_{batch_size}_lr_{LR}_flt_{features_loss_target}_arch_{arch_string}_{tag14}_{tag7}_{tag10}_{tag13}.pkl", "wb") as outFile:
    pickle.dump(permutation_dict, outFile)

loaders_signal = {}

print("Preparing signal dataloaders...")


loaders_signal["SVJ_1000_0p5_large"] = GraphDataLoader(
    [
        [signals["SVJ_1000_0p5"], f"{signal_name}_large"],
    ],
    window_to_load=[[0., 1.] for _ in range(1)],
    normalizer=normalizer,
    device=device,
)

# loaders_validation_signal = {}

# for key in signals.keys():

#     loaders_validation_signal[key] = GraphDataLoader(
#         [
#             [signals[key], f"{key}_large"],
#         ],
#         window_to_load=[[0., 1.] for _ in range(1)],
#         batch_size=batch_size,
#         normalizer=normalizer,
#         device=device,
#     )

print("Preparing model...")


import time
from tqdm import tqdm
import mdmm

if args.autoencoder == "architectures/GraphAE.py":
    from architectures.GraphAE import graphAE
elif args.autoencoder == "architectures/GraphAE_GIN.py":
    from architectures.GraphAE_GIN import graphAE


# Create the model
model = graphAE(
    num_features=12,
    architecture=architecture,
).to(device)


# print("Defining mdmm constraints...")

# # Constraining nodes loss
# constraint_node = mdmm.MaxConstraint(
#             lambda x, edges: model.features_loss(x, edges),
#             args.features_loss_target,
#             # damping=args.damping,
#             scale=70.,
#             )
# # Constraining edge loss
# constraint_edge = mdmm.MaxConstraint(
#             lambda x, edges, batch: model.edges_loss(x, edges, torch.tensor(batched_negative_sampling(edges, batch), dtype=torch.int64)),
#             args.features_loss_target,
#             # damping=args.damping,
#             scale=0.02,
#             )

# MDMM_module = mdmm.MDMM([constraint_node]).to(device)
# optimizer = MDMM_module.make_optimizer(model.parameters(), lr=LR, optimizer=torch.optim.NAdam)

print("Defining optimizer...")

# Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=LR)


from torch_geometric.utils import batched_negative_sampling, negative_sampling
import matplotlib.pyplot as plt
import numpy as np


print("Defining training and validation loops...")

# Define training function
def train_on_batch(model, batch, optimizer):
    """
    Train the model for one epoch.

    Args:
    - model (torch.nn.Module): the model to train
    - batch (torch_geometric.data.Data): the batch of data
    - optimizer (torch.optim.Optimizer): the optimizer to use

    Returns:
    - float: the edges loss
    - float: the features loss
    """

    model.train()
    x = batch.x
    edges = batch.edge_index
    _batch = batch.batch

    edges_loss = model.edges_loss(
        x,
        edges,
        # Negative sampling is important to also show the network
        # examples of non-connected nodes
        batched_negative_sampling(
            edges,
            _batch,
        ),
    )

    features_loss = model.features_loss(x, edges)

    # mdmm_return = MDMM_module(edges_loss, [(x, edges)])
    # mdmm_return = MDMM_module(features_loss, [(x, edges, _batch)])
                    
    optimizer.zero_grad()
    # mdmm_return.value.backward()
    edges_loss.backward()  
    # features_loss.backward()
    optimizer.step()

    # with torch.no_grad():
    #     features_loss = model.features_loss(x, edges)

    # with torch.no_grad():
    #         edges_loss = model.edges_loss(
    #             x,
    #             edges,
    #             torch.tensor(batched_negative_sampling(edges, _batch), dtype=torch.int64),
    #         )
    

    return edges_loss.detach().item(), features_loss.detach().item()
    # return float(edges_loss.detach()), float(features_loss.detach())


# Define validation function
def validate_on_graph(model, graph):
    """
    Validate the model for one graph.

    Args:
    - model (torch.nn.Module): the model to validate
    - graph: graph to pass to model

    Returns:
    - float: the edges loss
    - float: the features loss
    """

    model.eval()
    x = graph.x
    edges = graph.edge_index
    # _batch = batch.batch

    edges_loss = model.edges_loss(
        x,
        edges,
        # Negative sampling is important to also show the network
        # examples of non-connected nodes
        negative_sampling(
                edges,
            ).to(device),
    )

    features_loss = model.features_loss(x, edges)

    return float(edges_loss.detach()), float(features_loss.detach())

# Train the model

print("Preallocating loss lists and dictionaries...")

losses = []
losses_edges = []
losses_nodes = []
avg_train_losses = []
avg_train_edges_losses = []
avg_train_nodes_losses = []

val_losses = []
val_losses_edges = []
val_losses_nodes = []
avg_val_losses = []


val_sig_losses = {}
val_sig_losses_edges = {}
val_sig_losses_nodes = {}
avg_val_sig_losses = {}

signal_name = "SVJ_1000_0p5_large"
val_sig_losses[signal_name] = []
val_sig_losses_edges[signal_name] = []
val_sig_losses_nodes[signal_name] = []
avg_val_sig_losses[signal_name] = []


epochs = 400
# alpha_train = torch.tensor(alpha_train_val).to(device)
# alpha_test = torch.tensor(alpha_test_val).to(device)

from sklearn.metrics import roc_auc_score, roc_curve
from plotUtils import calc_auc

start = time.time()

mean_auc_per_epoch = []

# Training and validation loop

for epoch in range(epochs):

    val_losses_ = []
    val_sig_losses_ = {}
    
    for batch in tqdm(loader_training, desc=f"Epoch {epoch}", leave=False):
        loss_edges, loss_features = train_on_batch(model, batch, optimizer)

        losses.append((loss_edges+loss_features))
        losses_edges.append(loss_edges)
        losses_nodes.append((loss_features))
    
    avg_train_losses.append(np.mean(losses))
    avg_train_edges_losses.append(np.mean(losses_edges))
    avg_train_nodes_losses.append(np.mean(losses_nodes))

    experiment.log_metric("train loss", avg_train_losses[-1], epoch=epoch)
    experiment.log_metric("edges train loss", avg_train_edges_losses[-1], epoch=epoch)
    experiment.log_metric("nodes train loss", avg_train_nodes_losses[-1], epoch=epoch)

    for graph in tqdm(loader_validation, desc=f"Epoch {epoch}", leave=True):
        val_loss_edges, val_loss_features = validate_on_graph(model, graph)

        val_losses.append((val_loss_edges+val_loss_features))
        val_losses_.append((val_loss_edges+val_loss_features))
        val_losses_edges.append(val_loss_edges)
        val_losses_nodes.append((val_loss_features))
    
    avg_val_losses.append(np.mean(val_losses))
    experiment.log_metric("bkg validation loss", avg_val_losses[-1], epoch=epoch)

    if (epoch+1) % 20 == 0:
        torch.save(model.state_dict(), f"models/model_10_07_epoch_{epoch+1}_bs_{batch_size}_lr_{LR}_flt_{features_loss_target}_arch_{arch_string}_{tag14}_{tag7}_{tag10}_{tag13}.pt")

    signal_name = "SVJ_1000_0p5_large"
    val_sig_losses_[signal_name] = []
    val_sig_losses[signal_name] = []
    for graph in tqdm(loaders_signal["SVJ_1000_0p5_large"], desc=f"Signal Validation Epoch {epoch}, signal {signal_name}", leave=True):
        val_loss_edges, val_loss_features = validate_on_graph(model, graph)
        
        val_sig_losses[signal_name].append((val_loss_edges+val_loss_features))
        val_sig_losses_[signal_name].append((val_loss_edges+val_loss_features))
        val_sig_losses_edges[signal_name].append(val_loss_edges)
        val_sig_losses_nodes[signal_name].append((val_loss_features))

    _, mean_auc = calc_auc(val_sig_losses_, val_losses_)  
    mean_auc_per_epoch.append(mean_auc)  
    avg_val_sig_losses[signal_name].append(np.mean(val_sig_losses[signal_name]))

    experiment.log_metric("signal validation loss SVJ_1000_0p5 ", avg_val_sig_losses["SVJ_1000_0p5_large"][-1], epoch=epoch)
    experiment.log_metric("Mean auc", mean_auc_per_epoch[-1], epoch=epoch)



print(f"--> Training completed in {time.time()-start:.2f} seconds")

# Save the model
torch.save(model.state_dict(), f"models/model_10_07_final_bs_{batch_size}_lr_{LR}_flt_{features_loss_target}_arch_{arch_string}_{tag14}_{tag7}_{tag10}_{tag13}.pt")

# # Plot nodes and edges loss
# plt.plot(losses, label="Total loss", marker="o")
# plt.plot(losses_edges, label="Edges loss", marker="o")
# plt.plot(losses_nodes, label="Nodes loss", marker="o")

# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.yscale("log")
# plt.legend()
# # plt.show()
# plt.savefig(f"losses_per_batch_plots/19_06/losses_evolution_bs_{batch_size}_lr_{LR}_flt_{features_loss_target}_arch_{arch_string}_{tag14}_{tag3}_{tag5}_{tag8}.pdf")
# plt.close()


# # Plot training vs validation loss
# plt.plot(avg_train_losses, label="Train loss QCD", marker="o")
# plt.plot(avg_val_losses, label="Validation loss QCD", marker="o")
# plt.plot(avg_val_sig_losses[signal_name], label="Validation loss"+signal_name, marker="o")

# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.yscale("log")
# plt.legend()
# # plt.show()
# plt.savefig(f"losses_train_validation/19_06/losses_train_validation_bs_{batch_size}_lr_{LR}_flt_{features_loss_target}_arch_{arch_string}_{tag14}_{tag3}_{tag5}_{tag8}.pdf")
# plt.close()


# # Evaluate the model on the test set

# model.eval()

# losses_testing_qcd = []

# for graph in tqdm(loader_testing, desc="Testing QCD", leave=True):
#     x = graph.x.to(device)
#     edges = graph.edge_index.to(device)

#     edges_loss = model.edges_loss(
#         x,
#         edges,
#         negative_sampling(
#             edges,
#         ).to(device),
#     )

#     features_loss = model.features_loss(x, edges)
#     loss = edges_loss + features_loss

#     losses_testing_qcd.append(float(loss.detach()))



# losses_testing_signal = {}

# for signal_name, loader_signal in loaders_signal.items():
#     losses_testing_signal[signal_name] = []
#     for graph in tqdm(loader_signal, desc=f"Testing SVJ: {signal_name}", leave=True):
#         x = graph.x.to(device)
#         edges = graph.edge_index.to(device)

#         edges_loss = model.edges_loss(
#             x,
#             edges,
#             negative_sampling(
#                 edges,
#             ).to(device),
#         )

#         features_loss = model.features_loss(x, edges)
#         loss = edges_loss + features_loss

#         losses_testing_signal[signal_name].append(float(loss.detach()))



# for signal_name in losses_testing_signal.keys():
#     plt.hist(losses_testing_qcd, bins=50, histtype="step", label="QCD", density=True)
#     plt.hist(losses_testing_signal[signal_name], bins=50, histtype="step", label=signal_name, density=True)
#     plt.xlabel("Loss")
#     plt.ylabel("Density")
#     plt.legend()
#     # plt.show()

#     plt.savefig(f"losses_test_plots/19_06/losses_test_{signal_name}_bs_{batch_size}_lr_{LR}_flt_{features_loss_target}_arch_{arch_string}_{tag14}_{tag3}_{tag5}_{tag8}.pdf")
#     plt.close()


# from sklearn.metrics import roc_auc_score, roc_curve
# import copy
# import numpy as np

# # Create physical weights to use in auc curves 
# weights_bkg_auc = []

# qcd_for_weights = copy.deepcopy(qcd)                    # deep copies to avoid modifying the graph data
# # ttjets_for_weights = copy.deepcopy(ttjets)
# signals_for_weights = copy.deepcopy(signal_graphs)

# # Only create weights for graphs used for tesing
# qcd_for_weights = qcd_for_weights[int(0.7*len(qcd_for_weights)):int(1.0*len(qcd_for_weights))]
# weights_bkg_auc.append(qcd_for_weights.w.cpu().numpy())

# # ttjets_for_weights = ttjets_for_weights[int(0.7*len(ttjets_for_weights)):int(1.0*len(ttjets_for_weights))]
# # weights_bkg_auc.append(ttjets_for_weights.w.cpu().numpy())

# weights_bkg_auc_array = np.concatenate(weights_bkg_auc)

# # Store the signal weights in a dictionary
# weights_sig_auc = {}
# all_weights_auc = {}


# signals_for_weights = copy.deepcopy(signal_graphs)
# for signal_name, signal_data in signals_for_weights.items():
#     weights_sig_auc[signal_name] = signal_data.w.cpu().numpy()

#     weights_auc_this_sig = [weights_bkg_auc_array, weights_sig_auc[signal_name]]
#     all_weights_auc[signal_name] = np.concatenate(weights_auc_this_sig)


# # Make labels and scores dictionaries
# y_true = {}
# y_score = {}
# fpr = {}
# tpr = {}
# auc = {}

# for signal_name, signal_losses in losses_testing_signal.items():
#     y_true[signal_name] = [0]*len(losses_testing_qcd) + [1]*len(signal_losses)
#     y_score[signal_name] = losses_testing_qcd + signal_losses

#     fpr[signal_name], tpr[signal_name], _ = roc_curve(y_true[signal_name], y_score[signal_name], sample_weight=all_weights_auc[signal_name])
#     auc[signal_name] = roc_auc_score(y_true[signal_name], y_score[signal_name], sample_weight=all_weights_auc[signal_name])

# auc_list = list(auc.values())

# mean_auc = np.mean(auc_list)

# experiment.log_metric("mean auc", mean_auc)


# for signal_name in auc.keys():
#     plt.plot(fpr[signal_name], tpr[signal_name], label=f"AUC = {auc[signal_name]:.3f}")
#     plt.xlabel("False positive rate")
#     plt.ylabel("True positive rate")
#     plt.title(f"ROC curve for {signal_name}")
#     plt.legend()
#     # plt.show()
#     plt.savefig(f"ROC_curves/19_06/ROC_{signal_name}_bs_{batch_size}_lr_{LR}_flt_{features_loss_target}_arch_{arch_string}_{tag14}_{tag3}_{tag5}_{tag8}.pdf")
#     plt.close()


# import re

# def make_dict(auc):
#     dict_result = {}

#     for key in auc.keys():
#         pattern = r"SVJ_(\d+)_0p(\d+)"
#         match = re.search(pattern, key)
#         if match:
#             mMed = int(match.group(1))
#             rinv = float(f"0.{match.group(2)}")
#             sig_key = (mMed, rinv)
#             dict_result[sig_key] = auc[key]
#     return dict_result

# # Example usage:


# auc_dict = make_dict(auc)

# import plotUtils as plot

# plot.plot_table(auc_dict, "AUC", f"AUC_tables/19_06/AUC_table_bs_{batch_size}_lr_{LR}_flt_{features_loss_target}_arch_{arch_string}_{tag14}_{tag3}_{tag5}_{tag8}", "$m_{Med}$", "$r_{inv}$",)