import torch
import torch.nn.functional as F
from torch_geometric.utils import batched_negative_sampling, negative_sampling, to_dense_adj
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

from utils.JetGraphProducer import JetGraphProducer
from utils.LundTreeUtilities import OnTheFlyNormalizer
from utils.PlotUtils import plot_tree
from data_loading.graphDataLoader import GraphDataLoader
from architectures.LundNet import LundNet
from sklearn.metrics import roc_auc_score, roc_curve

torch.set_num_threads(8)

# **Step 1: Generate graphs from signal data**
signal_ggZH_ZToNuNu = JetGraphProducer(
    "/pnfs/psi.ch/cms/trivcat/store/user/ext-ebaldo/vh_bb_test_skims/2017/0_leptons_selection_Hbb_boosted/nominal/ggZH_HToBB_ZToNuNu/", 
    "ggZH_HToBB_ZToNuNu",
    n_store_jets=2,
    use_lund_decomp=True,
    n_lund_vars=5,
    save_energy_fractions=True,
    save_4vectors_in_lund_tree=True,
    weights="xsec",
    verbose=True,
    label=1,  # Label for signal graphs
)

signal_ggZH_ZToLL = JetGraphProducer(
    "/pnfs/psi.ch/cms/trivcat/store/user/ext-ebaldo/vh_bb_test_skims/2017/0_leptons_selection_Hbb_boosted/nominal/ggZH_HToBB_ZToLL/",
    "ggZH_HToBB_ZToLL",
    n_store_jets=2,
    use_lund_decomp=True,
    n_lund_vars=5,
    save_energy_fractions=True,
    save_4vectors_in_lund_tree=True,
    weights="xsec",
    verbose=True,
    label=1,  # Label for signal graphs
)

# **Produce the graphs for the other background datasets**
background_datasets = [
    "WJetsToLNu_Pt-250To400",
    "WJetsToLNu_Pt-400To600",
    #"WJetsToLNu_Pt-600ToInf"
    #"Z1JetsToNuNu_M-50_LHEFilterPtZ-250To400",
    #"Z1JetsToNuNu_M-50_LHEFilterPtZ-50To150",
    #"Z2JetsToNuNu_M-50_LHEFilterPtZ-400ToInf",
    #"Z1JetsToNuNu_M-50_LHEFilterPtZ-150To250",
    #"Z1JetsToNuNu_M-50_LHEFilterPtZ-400ToInf",
    #"Z2JetsToNuNu_M-50_LHEFilterPtZ-150To250",
    #"Z2JetsToNuNu_M-50_LHEFilterPtZ-50To150"
]

background_graphs = {}

for dataset in background_datasets:
    # Creazione del grafo per il singolo dataset
    graph = JetGraphProducer(
        f"/pnfs/psi.ch/cms/trivcat/store/user/ext-ebaldo/vh_bb_test_skims/2017/0_leptons_selection_Hbb_boosted/nominal/{dataset}/",  # Path to the dataset
        dataset,  
        n_store_jets=2,
        use_lund_decomp=True,
        n_lund_vars=5,
        save_energy_fractions=True,
        save_4vectors_in_lund_tree=True,
        weights="xsec",
        verbose=True,
        label=0,  
    )
    background_graphs[dataset] = graph  


# **Step 2: Compute normalization parameters (means and stds)**
means, stds = 0., 0.
weights, counts = [], []

print(background_datasets)

# Compute means
for dataset in background_datasets:
    for graph in tqdm(background_graphs[dataset],desc=f"calculating means for {dataset}"):
        means += graph.x.sum(dim=0) * graph.w
        weights.append(graph.w)
        counts.append(graph.num_nodes)

    means /= sum(torch.tensor(weights) * torch.tensor(counts))

print("means:")
print(means)

# Compute stds
for dataset in background_datasets:
    for graph in background_graphs[dataset]:
        stds += ((graph.x - means)**2).sum(dim=0) * graph.w

    stds /= sum(torch.tensor(weights) * torch.tensor(counts))
    stds = torch.sqrt(stds)

# Initialize normalizer
normalizer = OnTheFlyNormalizer(["x"], means, stds)


# **Step 3: Inspect the graph structure**
# print(background_graphs["WJetsToLNu_Pt-250To400"][0])  # Print details of the first graph

# print(background_graphs["WJetsToLNu_Pt-250To400"][0].x.shape)  # Print the shape of node features
# print(background_graphs["WJetsToLNu_Pt-250To400"][0].x)        # Print the node features

# # Print the adjacency matrix in dense format
# print(to_dense_adj(background_graphs["WJetsToLNu_Pt-250To400"][0].edge_index))

# # Print the sparse adjacency matrix
# print(background_graphs["WJetsToLNu_Pt-250To400"][0].edge_index.shape)
# print(background_graphs["WJetsToLNu_Pt-250To400"][0].edge_index)

# # **Optional: Plot a Lund graph**
# plot_tree(background_graphs["WJetsToLNu_Pt-250To400"][0], 4, "$p_T$", "myTree", show=True)


# **Step 4: Load the data for training and testing**
background_data_pairs = [
    [f"/pnfs/psi.ch/cms/trivcat/store/user/ext-ebaldo/vh_bb_test_skims/2017/0_leptons_selection_Hbb_boosted/nominal/{dataset}/", dataset]
    for dataset in background_datasets
]

loader_training = GraphDataLoader(
    background_data_pairs,
    window_to_load=[[0., 0.7] for _ in range(len(background_data_pairs))],
    batch_size=64,
    normalizer=normalizer,
)

loader_testing = GraphDataLoader(
    background_data_pairs,
    window_to_load=[[0.7, 1.] for _ in range(len(background_data_pairs))],
    batch_size=64,
    normalizer=normalizer,
)

loader_signal = GraphDataLoader(
    [
        ["/pnfs/psi.ch/cms/trivcat/store/user/ext-ebaldo/vh_bb_test_skims/2017/0_leptons_selection_Hbb_boosted/nominal/ggZH_HToBB_ZToNuNu/", 
         "ggZH_HToBB_ZToNuNu"],
        ["/pnfs/psi.ch/cms/trivcat/store/user/ext-ebaldo/vh_bb_test_skims/2017/0_leptons_selection_Hbb_boosted/nominal/ggZH_HToBB_ZToLL/", 
         "ggZH_HToBB_ZToLL"],
    ],
    window_to_load=[[0., 1.] for _ in range(2)],
    batch_size=64,
    normalizer=normalizer,
)


# **Step 5: Initialize the LundNet autoencoder**
model = LundNet(
    input_dims=12,                    # Number of node features (input dimensions)
    conv_params=[                    # Convolutional layers architecture
        (12, 128),                   # Each tuple is a layer (input, output) dimensions
        (128, 64),
        (64, 32),
    ],
    fc_params = [(32, 0.8)],
)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # Optimizer


# **Step 6: Define the training function**
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_on_batch(model, batch, optimizer, alpha=1.):
    """
    Train the model for one epoch.

    Args:
    - model (torch.nn.Module): the model to train
    - batch (torch_geometric.data.Data): the batch of data
    - optimizer (torch.optim.Optimizer): the optimizer to use
    - alpha (float): the weight of the features loss in the total loss

    Returns:
    - float: the edges loss
    - float: the features loss
    """
    model.train()

    # print("batch")
    # print(batch)

    # print(f"Batch size (number of graphs): {batch.num_graphs}")
    # print(f"Number of nodes in batch: {batch.x.size(0)}")
    # print(f"Edge count in batch: {batch.edge_index.size(1)}")

    # Extract data from the batch
    # x = batch.x.to(device)
    # edges = batch.edge_index.to(device)  #torch create edge_index according to the dataset 
    true_labels = batch.y.to(device)  # Ground truth labels for edges

    # Predict logits for edges (output before applying sigmoid)
    logits = model(batch) 

    # print(f"Logits shape: {logits.shape}, values: {logits}")
    # print(f"True labels shape: {true_labels.shape}, values: {true_labels}")
    
    logits = logits.squeeze(-1)
    
    # Compute Binary Cross Entropy Loss on edges
    loss = F.binary_cross_entropy_with_logits(logits, true_labels.float())

    # features_loss = model.features_loss(x, edges)
    # loss = edges_loss + alpha * features_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return float(loss.detach())


# **Step 7: Train the model**
losses, losses_edges, losses_nodes = [], [], []
epochs, alpha = 1, 150  # Use more epochs for actual training

start = time.time()
for epoch in range(epochs):
    for batch in tqdm(loader_training, desc=f"Epoch {epoch}", leave=False):
        loss_edges = train_on_batch(model, batch, optimizer, alpha)
        losses_edges.append(loss_edges)

print(f"--> Training completed in {time.time()-start:.2f} seconds")

# Plot training losses
plt.plot(losses_edges, label="Edges loss", marker="o")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.yscale("log")
plt.legend()
plt.savefig("/t3home/ext-ebaldo/SVJ_GNN/output/Training_Loss.png")


# **Step 8: Evaluate the model**
model.eval()

losses_testing_background = []

for graph in tqdm(loader_testing, desc="Testing Background", leave=True):
    x = graph.x
    edges = graph.edge_index

    edges_loss = train_on_batch(model, graph, optimizer, alpha=1.)

    # features_loss = model.features_loss(x, edges)
    # loss = edges_loss + alpha*features_loss

    losses_testing_background.append(float(edges_loss))

losses_testing_signal = []

for graph in tqdm(loader_signal, desc="Testing Signal", leave=True):
    x = graph.x
    edges = graph.edge_index

    edges_loss = train_on_batch(model, graph, optimizer, alpha=1.)

    # features_loss = model.features_loss(x, edges)
    # loss = edges_loss + alpha * features_loss

    losses_testing_signal.append(float(edges_loss))


# **Step 9: Plot the results**
# Plot loss distributions
plt.clf()
plt.hist(losses_testing_signal, bins=50, histtype="step", label="Background", density=True)
plt.hist(losses_testing_background, bins=50, histtype="step", label="Signal", density=True)
plt.xlabel("Loss")
plt.ylabel("Density")
plt.legend()
plt.savefig("/t3home/ext-ebaldo/SVJ_GNN/output/Loss_Distribution.png")

