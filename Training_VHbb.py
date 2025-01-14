import torch
import torch.nn.functional as F
from torch_geometric.utils import batched_negative_sampling, negative_sampling, to_dense_adj
import numpy as np
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
signal_ToNuNu_graph = JetGraphProducer(
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

signal_ToLL_graph = JetGraphProducer(
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
    "WJetsToLNu_Pt-600ToInf"
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

# **Step 2: Compute normalization parameters (means and stds) for all datasets combined (background + signal)**

all_graphs = list(background_graphs.values()) + [signal_ToNuNu_graph, signal_ToLL_graph]

means, stds = 0., 0.
weights, counts = [], []

# mean
for graph in all_graphs:
    for g in tqdm(graph, desc="Calculating means for all datasets"):
        means += g.x.sum(dim=0) * g.w  # Somma ponderata delle caratteristiche
        weights.append(g.w)
        counts.append(g.num_nodes)

# Normalize the sum by the total number of weighted nodes
means /= sum(torch.tensor(weights) * torch.tensor(counts))

print("Combined means (background + signal):")
print(means)

# stds
for graph in all_graphs:
    for g in graph:
        stds += ((g.x - means)**2).sum(dim=0) * g.w  # Differenze quadrate ponderate

# Normalize the sum by the total number of weighted nodes
stds /= sum(torch.tensor(weights) * torch.tensor(counts))
stds = torch.sqrt(stds)

print("Combined stds (background + signal):")
print(stds)

normalizer = OnTheFlyNormalizer(["x"], means, stds)



# **Step 3: Inspect the graph structure**


# **Step 4: Load the data for training and testing**
background_data_pairs = [
    [f"/pnfs/psi.ch/cms/trivcat/store/user/ext-ebaldo/vh_bb_test_skims/2017/0_leptons_selection_Hbb_boosted/nominal/{dataset}/", dataset]
    for dataset in background_datasets
]

signal_data_pairs = [
    [f"/pnfs/psi.ch/cms/trivcat/store/user/ext-ebaldo/vh_bb_test_skims/2017/0_leptons_selection_Hbb_boosted/nominal/ggZH_HToBB_ZToNuNu/", 
     "ggZH_HToBB_ZToNuNu"],
    [f"/pnfs/psi.ch/cms/trivcat/store/user/ext-ebaldo/vh_bb_test_skims/2017/0_leptons_selection_Hbb_boosted/nominal/ggZH_HToBB_ZToLL/", 
     "ggZH_HToBB_ZToLL"]
]

all_data_pairs = background_data_pairs + signal_data_pairs

signal_total_weight = (
    signal_ToNuNu_graph.w.sum() +
    signal_ToLL_graph.w.sum()
)

background_total_weight = sum(graph.w.sum() for graph in background_graphs.values())

# Frazione totale (0.5 per segnali, 0.5 per background)
dataset_fractions = []
for graph in background_graphs.values():
    dataset_fractions.append(0.5 * (graph.w.sum() / background_total_weight))  # Normalizzato per il background

dataset_fractions.append(0.5 * (signal_ToNuNu_graph.w.sum() / signal_total_weight))  # Segnali
dataset_fractions.append(0.5 * (signal_ToLL_graph.w.sum() / signal_total_weight))


loader_training = GraphDataLoader(
    all_data_pairs,
    window_to_load=[[0., 0.7] for _ in range(len(all_data_pairs))],
    batch_size=64,
    normalizer=normalizer,
    dataset_fraction=dataset_fractions,
)

loader_testing = GraphDataLoader(
    all_data_pairs,
    window_to_load=[[0.7, 1.] for _ in range(len(all_data_pairs))],
    batch_size=64,
    normalizer=normalizer,
    dataset_fraction=dataset_fractions,
)


# **Step 5: Initialize the LundNet 
model = LundNet(
    input_dims=12,                    # Number of node features (input dimensions)
    conv_params=[
        (12, 256),                      # Convolutional layers architecture
        (256, 128),
        (128, 64),
        (64, 32),
    ],
    fc_params = [(32, 0.8)],
)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # Optimizer


# **Step 6: Define the training function**
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_on_batch(model, batch, optimizer, alpha):

    model.train() #Training of the Network

    true_labels = batch.y.to(device)  # Ground truth labels for edges
    logits = model(batch) 
    logits = logits.squeeze(-1) #remove a dimension to have the same size as true_labels
    # Compute Binary Cross Entropy Loss on edges
    loss = F.binary_cross_entropy_with_logits(logits, true_labels.float())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return float(loss.detach())

def validate_on_batch(model, batch):

    model.eval() # Validation

    with torch.no_grad(): 
        true_labels = batch.y.to(device)  
        logits = model(batch)
        logits = logits.squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits, true_labels.float())
    return float(loss)


# Step 7: Train the model
epochs = 5
alpha = 150  

epoch_training_losses = []  
epoch_validation_losses = []  

start = time.time()
for epoch in range(epochs):
    # Training loop
    batch_training_losses = [] 
    for batch in tqdm(loader_training, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
        BCE_loss = train_on_batch(model, batch, optimizer, alpha)  
        batch_training_losses.append(BCE_loss)  
    
    avg_training_loss = sum(batch_training_losses) / len(batch_training_losses)
    epoch_training_losses.append(avg_training_loss)
    
    # Validation loop
    batch_validation_losses = [] 
    with torch.no_grad():  
        for batch in loader_testing:
            validation_loss = validate_on_batch(model, batch)  
            batch_validation_losses.append(validation_loss)
    
    avg_validation_loss = sum(batch_validation_losses) / len(batch_validation_losses)
    epoch_validation_losses.append(avg_validation_loss)
    
    # Print training and validation loss
    print(f"Epoch {epoch+1}: Training loss = {avg_training_loss:.4f}, Validation loss = {avg_validation_loss:.4f}")

print(f"--> Training completed in {time.time()-start:.2f} seconds") 

# Plot delle perdite rispetto al numero di epoche
plt.plot(range(1, epochs+1), epoch_training_losses, label="Training Loss", marker="o")
plt.plot(range(1, epochs+1), epoch_validation_losses, label="Validation Loss", marker="o")
plt.xlabel("Epoch")
plt.ylabel("Average Loss")
plt.yscale("log") 
plt.legend()
plt.savefig("/t3home/ext-ebaldo/SVJ_GNN/output/Epoch_5_Losses.png")



# **Step 8: Evaluate the model**
model.eval()

scores_testing_background = []
scores_testing_signal = []

for graph in loader_testing:
    logits = model(graph).squeeze(-1)
    scores = torch.sigmoid(logits).detach().cpu().numpy()  # apply sigmoid to logits
    labels = graph.y.cpu().numpy()  # 0 = background, 1 = signal

    # Separate the scores based on the labels
    scores_testing_background.extend(scores[labels == 0]) 
    scores_testing_signal.extend(scores[labels == 1])      

all_scores = np.concatenate([scores_testing_signal, scores_testing_background])
all_labels = np.concatenate([np.ones_like(scores_testing_signal), np.zeros_like(scores_testing_background)])
roc_auc = roc_auc_score(all_labels, all_scores)
print(f"ROC AUC: {roc_auc:.4f}") 

# **Step 9: Plot the results**

plt.clf()
plt.hist(scores_testing_signal, bins=50, histtype="step", label="Signal", density=True, color='blue')
plt.hist(scores_testing_background, bins=50, histtype="step", label="Background", density=True, color='red')
plt.xlabel("Score")
plt.ylabel("Density")
plt.legend()
plt.title("Score Distribution: Signal vs Background")
plt.savefig("/t3home/ext-ebaldo/SVJ_GNN/output/Score_Distribution_epoch_5.png")



