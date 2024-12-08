from comet_ml import Experiment
import numpy as np
from svj_tools.utils.JetGraphProducer import JetGraphProducer
from svj_tools.utils.LundTreeUtilities import OnTheFlyNormalizer
import torch
from svj_tools.data_loading.graphDataLoader import GraphDataLoader
from svj_tools.architectures.GraphAE import graphAE
import time
from tqdm import tqdm
from torch_geometric.utils import batched_negative_sampling, negative_sampling
import matplotlib.pyplot as plt
import argparse
from sklearn.metrics import roc_auc_score, roc_curve
import os
from datetime import datetime
import random
import pickle
import socket

current_date = datetime.now().strftime("%m-%d_%H-%M")

# If you run on the PSI cluster, limit the number of threads to 8
# Get the current hostname
hostname = socket.gethostname()
print(hostname)

# Define the condition to check if running on the PSI T3 cluster
if "t3ui03" in hostname.lower() or "t3ui02" in hostname.lower() or "t3ui01" in hostname.lower():
    torch.set_num_threads(8)
    print("Running on PSI T3 cluster: limiting threads to 8")
else:
    print("Not running on PSI T3 cluster: using default thread settings")
    
####################################################################################################
# Import parameters from argparse
# Setup Comet
# Create output directory

parser = argparse.ArgumentParser(description="Training script")

parser.add_argument('--epoch', type=int, required=True, help="Number of epochs")
parser.add_argument('--batch_size', type=int, required=True, help="Batch size")
parser.add_argument('--learning_rate', type=float, required=True, help="Learning rate")
parser.add_argument('--nlund', type=int, required=True, help="Nlund value")
parser.add_argument('--architecture', nargs='+', type=int, required=True, help="Neural network architecture")
parser.add_argument('--output_dir', type=str, required=True, help="Output directory")
parser.add_argument('--alpha', type=float, required=True, help="Alpha value")
parser.add_argument('--test_alpha', type=float, required=True, help="Test alpha value")

args = parser.parse_args()

if args.alpha == 0 and args.test_alpha != 0:
    print("WARNING: alpha is set to 0, but test_alpha is not. This will result in a model that is not trained on the features loss, but will be tested on it. This is not recommended.")


num_features = 12

# Define Hyperparameters for Comet
hyperparams = {"epochs": args.epoch,
               "batch_size": args.batch_size,
               "learning_rate": args.learning_rate, 
               "nlund": args.nlund,
               "architecture": args.architecture,
               "alpha": args.alpha,
               "test_alpha": args.test_alpha,
               "num_features": num_features}

# Create the Comet experiment
experiment = Experiment(
  api_key="HwVMyj07I6zmVlQTSXkzBqvsn",
  project_name="Test",
  workspace="wiegerinho"
)

experiment.log_parameters(hyperparams)

arch_str = "_".join(map(str, args.architecture))

output_subdir = f"{current_date}_{arch_str}_n_features_{num_features}_numepochs_{args.epoch}"
output_path = os.path.join(args.output_dir, output_subdir)
os.makedirs(output_path, exist_ok=True)


####################################################################################################

# Produce the graphs. The call to JetGraphProducer will produce the graphs and
# save them in the specified directory if it is not already present.

qcd_paths = "/pnfs/psi.ch/cms/trivcat/store/user/ewiegers/Data/QCD/"


qcd = {
    "470to600GeV": {"path": qcd_paths + "qcd_pt-470to600GeV"},
    "600to800GeV": {"path": qcd_paths + "qcd_pt-600to800GeV"},
    "800to1000GeV": {"path": qcd_paths + "qcd_pt-800to1000GeV"},
    "1000to1400GeV": {"path": qcd_paths + "qcd_pt-1000to1400GeV"},
    "1400to1800GeV": {"path": qcd_paths + "qcd_pt-1400to1800GeV"},
    "1800to2400GeV": {"path": qcd_paths + "qcd_pt-1800to2400GeV"},
    "2400to3200GeV": {"path": qcd_paths + "qcd_pt-2400to3200GeV"},
    "3200toInfGeV": {"path": qcd_paths + "qcd_pt-3200toInfGeV"},
}

qcd_graphs = {}


for qcd_name, config in qcd.items():
    # Count the number of ROOT files in the directory
    n_root_files = len([f for f in os.listdir(config["path"]) if f.endswith(".root")])
    
    # Calculate the number of events per file as 10000 divided by the number of ROOT files
    n_events = 10000 // n_root_files if n_root_files > 0 else 0

    qcd_graphs[qcd_name] = JetGraphProducer(
        root=config["path"],
        output_dataset_name=qcd_name + "_medium",
        n_store_jets=2,
        max_events_to_process=n_events,
        use_lund_decomp=True,
        n_lund_vars=5,
        save_energy_fractions=True,
        save_4vectors_in_lund_tree=True,
        weights="xsec",
        verbose=True,
        label=0,
        use_displacement=False,
    )


TTJets_paths = "/pnfs/psi.ch/cms/trivcat/store/user/ewiegers/Data/TTJets/"

ttbar = {
    "TTJets_600to800GeV": {"path": TTJets_paths + "TTJets_ht-600to800GeV"},
    "TTJets_800to1200GeV": {"path": TTJets_paths + "TTJets_ht-800to1200GeV"},
    "TTJets_1200to2500GeV": {"path": TTJets_paths + "TTJets_ht-1200to2500GeV"},
    "TTJets_2500toInfGeV": {"path": TTJets_paths + "TTJets_ht-2500toInfGeV"},
}

ttbar_graphs = {}

for ttbar_name, config in ttbar.items():
    # Count the number of ROOT files in the directory
    n_root_files = len([f for f in os.listdir(config["path"]) if f.endswith(".root")])
    
    # Calculate the number of events per file as 10000 divided by the number of ROOT files
    n_events = 8000 // n_root_files if n_root_files > 0 else 0

    ttbar_graphs[ttbar_name] = JetGraphProducer(
        root=config["path"],
        output_dataset_name=ttbar_name + "_medium",
        n_store_jets=2,
        max_events_to_process=n_events,
        use_lund_decomp=True,
        n_lund_vars=5,
        save_energy_fractions=True,
        save_4vectors_in_lund_tree=True,
        weights="xsec",
        verbose=True,
        label=0,
        use_displacement=False,
    )



SVJ_path = "/pnfs/psi.ch/cms/trivcat/store/user/ewiegers/Data/"

signal={
    "SVJ_1500_0p3": SVJ_path + "sChannel_mZprime-1500GeV_mDark-20GeV_rInv-0p3_alpha-peak",
    "SVJ_4000_0p3": SVJ_path + "sChannel_mZprime-4000GeV_mDark-20GeV_rInv-0p3_alpha-peak",
    "SVJ_4000_0p5": SVJ_path + "sChannel_mZprime-4000GeV_mDark-20GeV_rInv-0p5_alpha-peak",
    "SVJ_4000_0p7": SVJ_path + "sChannel_mZprime-4000GeV_mDark-20GeV_rInv-0p7_alpha-peak",
}


signal_graphs = {}


for signal_name, signal_path in signal.items():
    signal_graphs[signal_name] = JetGraphProducer(
        signal_path,
        signal_name + "_large",
        n_store_jets = 2,
        use_lund_decomp=True,
        n_lund_vars=5,
        save_energy_fractions=True,
        save_4vectors_in_lund_tree=True,
        weights="xsec",
        verbose=True,
        label=0,
        use_displacement=False,
    )

####################################################################################################

# Compute the means and stds for the normalizer (if required)


# Initialize means and stds as tensors
means, stds = 0.0, 0.0
weights = []
num_nodes = []

# Calculate means across all datasets
for qcd_name, dataset in qcd_graphs.items():
    for graph in tqdm(dataset, desc=f"Calculating means for {qcd_name}"):
        means += graph.x.sum(dim=0) * graph.w
        weights.append(graph.w)
        num_nodes.append(graph.num_nodes)
for ttbar_name, dataset in ttbar_graphs.items():
    for graph in tqdm(dataset, desc=f"Calculating means for {ttbar_name}"):
        means += graph.x.sum(dim=0) * graph.w
        weights.append(graph.w)
        num_nodes.append(graph.num_nodes)

# Convert weights and num_nodes lists to tensors
weights = torch.tensor(weights)
num_nodes = torch.tensor(num_nodes)

# Calculate the weighted mean
means /= (weights * num_nodes).sum()

# Calculate standard deviations across all datasets
for qcd_name, dataset in qcd_graphs.items():
    for graph in tqdm(dataset, desc=f"Calculating stds for {qcd_name}"):
        stds += ((graph.x - means) ** 2).sum(dim=0) * graph.w
for ttbar_name, dataset in ttbar_graphs.items():
    for graph in tqdm(dataset, desc=f"Calculating stds for {ttbar_name}"):
        stds += ((graph.x - means) ** 2).sum(dim=0) * graph.w


# Normalize and take square root for the final standard deviation
stds /= (weights * num_nodes).sum()
stds = torch.sqrt(stds)

# `means` and `stds` now contain the weighted mean and std for all datasets

# Instantiate normalizer using loaded or calculated means and stds
normalizer = OnTheFlyNormalizer(["x"], means, stds)



####################################################################################################
# Load in the Training and Test Data



# Combine QCD and TTJets datasets
combined_datasets = [
    [config["path"], f"{qcd_name}_medium"] for qcd_name, config in qcd.items()
] + [
    [config["path"], f"{ttbar_name}_medium"] for ttbar_name, config in ttbar.items()
]

print("Start loading training data")

# Create loader with combined datasets
loader_training = GraphDataLoader(
    combined_datasets,
    window_to_load=[[0., 0.9] for _ in range(len(combined_datasets))],
    shuffle=True,
    batch_size=args.batch_size,
    normalizer=normalizer,
)

permutation = loader_training.get_permutation()

# Save the permutation to a pickle file
permutation_file = os.path.join(output_path, "permutation.pkl")
with open(permutation_file, "wb") as f:
    pickle.dump(permutation, f)

print(f"Permutation saved to {permutation_file}")
print("End loading training data")

print("Start loading testing data")

loader_testing = GraphDataLoader(
    combined_datasets,
    window_to_load=[[0.9, 1] for _ in range(len(combined_datasets))],
    shuffle = True,
    normalizer=normalizer,
    permutation=permutation,
)


# Reduced test loader to calculate ROC in every epoch
loader_testing_roc = GraphDataLoader(
    combined_datasets,
    window_to_load=[[0.9, 0.91] for _ in range(len(combined_datasets))],
    shuffle = True,
    normalizer=normalizer,
    permutation=permutation,
)

print("End loading testing data")

loader_svjs={}

for signal_name, signal_path in signal.items():
    loader_svjs[signal_name] = GraphDataLoader(
        [
            [signal_path, signal_name + "_large"]
        ],
        window_to_load=[[0., 1.] for _ in range(1)], 
        shuffle = False,
        normalizer=normalizer,
        permutation=None,
    )


loader_svj_roc={}

# Reduced signal loader to calculate ROC in every epoch
for signal_name, signal_path in signal.items():
    loader_svj_roc[signal_name] = GraphDataLoader(
        [
            [signal_path, signal_name + "_large"]
        ],
        window_to_load=[[0., .1] for _ in range(1)], 
        shuffle = False,
        normalizer=normalizer,
        permutation=None,
    )

# Check the normalization
features_background = [] 

for batch in tqdm(loader_training, desc="Extracting background features to check normalization"):
    for i in range(len(batch)):
        x = batch[i].x.cpu().numpy()
        if np.isnan(x).any():
            print(f"NaN found in batch {i}")
        features_background.append(x)

features_background = np.vstack(features_background)

print("Means of the background features:")
print(np.mean(features_background, axis=0))
print("Stds of the background features:")
print(np.std(features_background, axis=0))



####################################################################################################


# Create the model
model = graphAE(
    num_features=num_features,
    architecture=tuple(args.architecture)
)

optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)


# define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


####################################################################################################

# Define the training loop


def train_on_batch(model, batch, optimizer, alpha):
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
    x = batch.x
    edges = batch.edge_index
    _batch = batch.batch.to(device)

    edges_loss = model.edges_loss(
        x,
        edges,
        batched_negative_sampling(
            edges,
            _batch,
        ).to(device),
    )

    features_loss = model.features_loss(x, edges)
    loss = edges_loss + alpha*features_loss

    
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return float(edges_loss.detach()), float(features_loss.detach())

####################################################################################################

#Make lists to store losses
losses = []
losses_edges = []
losses_nodes = []
epochs = args.epoch
validation_losses = []
train_loss_list = []
start = time.time()


####################################################################################################

# Function to evaluate the model on the validation set (only necessary if you want to calculate the validation loss)

def evaluate_on_validation(model, loader_validation):
    model.eval()  # Set the model to evaluation mode
    losses_validation = []

    with torch.no_grad():  # Disable gradient computation for validation
        for batch in tqdm(loader_validation,desc="Validation", leave=False):
            x = batch.x
            edges = batch.edge_index

            edges_loss = model.edges_loss(
                x,
                edges,
                negative_sampling(
                    edges,
                ).to(device),
            )

            features_loss = model.features_loss(x, edges)
            loss = edges_loss +  args.test_alpha*features_loss

            losses_validation.append(float(loss.detach()))
            
    return np.mean(losses_validation) 


####################################################################################################

#Training Loop

print("Training started")
for epoch in range(epochs):
    # Training phase
    model.train()
    epoch_train_loss = 0
    num_batches = 0

    for batch in tqdm(loader_training, desc=f"Epoch {epoch}", leave=False):
        loss_edges, loss_features = train_on_batch(model, batch, optimizer, alpha=args.alpha)
        epoch_train_loss += loss_edges + args.alpha * loss_features
        num_batches += 1
        experiment.log_metric("loss_edges", loss_edges, step=num_batches)
        experiment.log_metric("loss_features", loss_features, step=num_batches)

    avg_train_loss = epoch_train_loss / num_batches
    train_loss_list.append(avg_train_loss)
    
    # Log average training loss to Comet
    experiment.log_metric("avg_train_loss", avg_train_loss, step=epoch)

    # Validation phase for ROC calculation
    model.eval()
    losses_validation_qcd = []
    print("Start calculating ROC of epoch" + str(epoch))
    # Calculate losses for QCD validation data
    for graph in tqdm(loader_testing_roc, desc="Validation QCD", leave=False):
        np.random.seed(0)
        random.seed(0)
        torch.manual_seed(0)
        x = graph.x
        edges = graph.edge_index
        edges_loss = model.edges_loss(
            x,
            edges,
            negative_sampling(edges).to(device)
        )
        features_loss = model.features_loss(x, edges)
        loss = edges_loss + args.test_alpha * features_loss
        losses_validation_qcd.append(float(loss.detach()))

    # Calculate losses for each SVJ signal validation data
    losses_validation_signal = {}
    for name, loader_signal in loader_svj_roc.items():
        losses_validation_signal[name] = []
        for graph in tqdm(loader_signal, desc=f"Validation SVJ: {name}", leave=False):
            np.random.seed(0)
            random.seed(0)
            torch.manual_seed(0)
            x = graph.x
            edges = graph.edge_index
            edges_loss = model.edges_loss(
                x,
                edges,
                negative_sampling(edges).to(device)
            )
            features_loss = model.features_loss(x, edges)
            loss = edges_loss + args.test_alpha * features_loss
            losses_validation_signal[name].append(float(loss.detach()))

    # Calculate ROC AUC for each signal
    for signal_name, signal_losses in losses_validation_signal.items():
        y_true = [0] * len(losses_validation_qcd) + [1] * len(signal_losses)
        y_score = losses_validation_qcd + signal_losses

        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc_score = roc_auc_score(y_true, y_score)

        # Log ROC metrics to Comet
        experiment.log_metric(f"{signal_name}_AUC", auc_score, step=epoch)

    print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {avg_train_loss:.4f}")
    

print(f"--> Training completed in {time.time() - start:.2f} seconds")


####################################################################################################
#Testing Procedure

# Save the model
model_path = os.path.join(output_path, f"model_{arch_str}.pt")
torch.save(model, model_path)


# Evaluate the model on the testing set
model.eval()

losses_testing_qcd = []

for graph in tqdm(loader_testing, desc="Testing QCD", leave=True):
    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)
    x = graph.x
    edges = graph.edge_index

    edges_loss = model.edges_loss(
        x,
        edges,
        negative_sampling(
            edges,
        ).to(device),
    )

    features_loss = model.features_loss(x, edges)
    loss = edges_loss + args.test_alpha * features_loss

    losses_testing_qcd.append(float(loss.detach()))

losses_testing_signal = {}

for name, loader_signal in loader_svjs.items():
    losses_testing_signal[name] = []
    for graph in tqdm(loader_signal, desc=f"Testing SVJ: {name}", leave=True):
        np.random.seed(0)
        random.seed(0)
        torch.manual_seed(0)
        x = graph.x
        edges = graph.edge_index

        edges_loss = model.edges_loss(
            x,
            edges,
            negative_sampling(
                edges,
            ).to(device),
        )

        features_loss = model.features_loss(x, edges)
        loss = edges_loss + args.test_alpha * features_loss

        losses_testing_signal[name].append(float(loss.detach()))


####################################################################################################

#Make plots and save them

plt.hist(losses_testing_qcd, bins=50, histtype="step", label="Background", density=True)
for signal_name in losses_testing_signal.keys():
    plt.hist(losses_testing_signal[signal_name], bins=50, histtype="step", label=signal_name, density=True)
plt.xlabel("Loss")
plt.ylabel("Density")
plt.legend()
plt.savefig(f"{output_path}/losses_{arch_str}.png")
plt.clf()

y_true={}
y_score={}
fpr={}
tpr={}
aucx={}

for signal_name, signal_losses in losses_testing_signal.items():
     y_true[signal_name] = [0]*len(losses_testing_qcd) + [1]*len(signal_losses)
     y_score[signal_name] = losses_testing_qcd + signal_losses
     fpr[signal_name], tpr[signal_name], _ = roc_curve(y_true[signal_name], y_score[signal_name]) 
     aucx[signal_name] = roc_auc_score(y_true[signal_name], y_score[signal_name])


plt.figure(figsize=(10, 7))

for signal_name in aucx.keys():
    plt.plot(fpr[signal_name], tpr[signal_name], label=f" {signal_name} AUC = {aucx[signal_name]:.3f}")

# Set the labels and title outside the loop
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title(f"ROC Curve for All Signals, Mean AUC: {np.mean(list(aucx.values()))}")
plt.legend()

# Save the combined plot to a file
plt.savefig(f"{output_path}/roc_curve_{arch_str}.png")

