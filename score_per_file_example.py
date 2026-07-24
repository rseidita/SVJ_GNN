import argparse
import pickle

import torch

from utils.ScoreProducer import ScoreProducer

parser = argparse.ArgumentParser()

parser.add_argument('--file', type=str, required=True, help='File to read the data from')
parser.add_argument('--output_folder', type=str, required=True, help="Output to store the files with the scores")
parser.add_argument(
    "--alpha",
    type=float,
    help="Weight for the features loss"
)

args = parser.parse_args()

input_file = args.file
output_folder = args.output_folder
alpha = args.alpha


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torch.load("/t3home/ewiegers/SVJ_GNN/Plots/12-12_12-39_1024_512_256_128_64_32_16_n_features_14_numepochs_30/model_1024_512_256_128_64_32_16.pkl")

# Load the normalizer
with open("/t3home/ewiegers/SVJ_GNN/Plots/12-12_12-39_1024_512_256_128_64_32_16_n_features_14_numepochs_30/normalizer.pkl", "rb") as f:
    normalizer = pickle.load(f)

# Define QCD datasets with paths

print("Starting the ScoreProducer")

ScoreProducer(root_file=input_file,
                use_displacement=True,
                n_lund_vars=5,
                save_energy_fractions=True,
                save_4vectors_in_lund_tree=True,
                weights="xsec",
                verbose=True,
                model = model,
                normalizer = normalizer,
                output_folder=output_folder,
                alpha = alpha)
