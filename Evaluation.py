import time
import utils.utils_evaluation as et
import torch
import os
from architectures.LundNet import LundNet
from utils.LundTreeUtilities import OnTheFlyNormalizer
import pickle
import argparse
from utils.utils_training import get_model_parameters

def main(name, suffix, batch_size, window_to_load, model_type, eval_type, jet_idx):
        
    base_directory_bkg = "/pnfs/psi.ch/cms/trivcat/store/user/rhampp/forTraining/Preselection_applied/Bkg/"
    base_directory_SVJ = "/pnfs/psi.ch/cms/trivcat/store/user/rhampp/forTraining/Preselection_applied/SVJ_new/"


    print("==> Evaluate Model...")
    start_eva = time.time()

    #Load model
    filename = "/" + name + ".pt"
    dict = torch.load("/t3home/rhampp/SVJ_GNN/output/"+ name +"/" + filename,  map_location=torch.device('cpu'))
    nLund_variables, energy_fractions, four_vectors = get_model_parameters(model_type)

    if energy_fractions:
        nLund_variables += 4
 
    # Create the model
    model = LundNet(
        conv_params = [[32, 32], [32, 32], [64, 64], [64, 64], [128, 128], [128, 128]],
        fc_params = [(128, 0.8)],
        input_dims = nLund_variables,
        add_fractions_to_lund=four_vectors
    )

    model.load_state_dict(dict["model_state_dict"])

    # Load normalizer
    with open("/t3home/rhampp/SVJ_GNN/output/"+name+"/Normalization.pkl",  'rb') as file:
        norm_dict = pickle.load(file)
    means = norm_dict["means"]
    stds = norm_dict["stds"]
    normalizer =  OnTheFlyNormalizer(["x"], means, stds)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    

    evaluation_dict = et.Evaluate_model(base_directory_bkg, base_directory_SVJ, model_type, eval_type, batch_size, normalizer, model, device, window_to_load, jet_idx)

    # compute roc and auc, safe them also in pickels
    roc, auc = et.roc_auc(evaluation_dict, name, suffix)
    et.plot_auc(auc, name, suffix)
    et.plot_roc(roc, name, suffix)
    et.plot_scores(evaluation_dict, name, suffix)
    et.plot_sic(roc, name, suffix)
    et.plot_dnn(evaluation_dict, name, suffix)

    print(f"==> Evaluation completed in {time.time()-start_eva:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--batch",
        default=64,
        help="Batch size",
        type=int,
        )
    parser.add_argument(
        "--name",
        default="LundNet",
        help="Name of the training (directory)",
        type=str,
        )
    parser.add_argument(
        "--suffix",
        default="eval",
        help="Suffix to say on which part it is evaluated",
        type=str,
        )
    
    parser.add_argument(
        "--window_to_load",
        default=[0.7,1],
        nargs='+',  
        type=float,
        help="Window to load for evaluation",
        )
    
    parser.add_argument(
        "--model",
        default="LN3",
        help="Which model to train: LN3, LN3p, LN5, LN5p",
        type=str,
    )

    parser.add_argument(
        "--jet_idx",
        default=None,
        help="Which jet to look at: 0=leading jet, 1=subleading jet, 2=subsubleading jet, None = all graphs in the dataset",
        type=int,
    )

    parser.add_argument(
        "--eval_type",
        default="",
        help="Which pt file to evaluate on: _3J(each jet single)",
        type=str,
    )
    
    args = parser.parse_args()

    batch_size = args.batch
    name = args.name
    window_to_load = args.window_to_load
    suffix = args.suffix
    model_type = args.model
    jet_idx = args.jet_idx
    eval_type = args.eval_type



    main(name, suffix, batch_size, window_to_load, model_type, eval_type, jet_idx)
