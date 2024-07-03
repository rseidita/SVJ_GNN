from architectures.LundNet import LundNet
from utils.LundTreeUtilities import OnTheFlyNormalizer
from data_loading.graphDataLoader_tagger import GraphDataLoader_tagger
from utils.utils_training import get_model_parameters
from utils.utils_tagJet import get_threshold_from_sic

import uproot4Utilities as uproot4Utl

import torch
import subprocess
import pickle
import argparse
import numpy as np
import os
import uproot
import time


def main(name, model_type):

    """Creates files with a branch FatJet_SVJtagger which indicates whether the fatjet is a SVJ or not
    This version handles reoccuring event numbers 
    Args:
        name(str): Name of the saved network 
        model_type(str): What kind of model the network is

    Return:

    """
    start = time.time()
        
    base_directory_SVJ = "/pnfs/psi.ch/cms/trivcat/store/user/rhampp/forTraining/Preselection_applied/SVJ_new/"
    base_directory_QCD = "/pnfs/psi.ch/cms/trivcat/store/user/rhampp/forTraining/Preselection_applied/Bkg/QCD_new/"

    filename = "/" + name + ".pt"
    dict = torch.load("/t3home/rhampp/SVJ_GNN/output/"+ name +"/" + filename,  map_location=torch.device('cpu'))
    nLund_variables, energy_fractions, four_vectors = get_model_parameters(model_type)

    if energy_fractions:
        nLund_variables += 4
 
    # Create model and load it
    model = LundNet(
        conv_params = [[32, 32], [32, 32], [64, 64], [64, 64], [128, 128], [128, 128]],
        fc_params = [(128, 0.8)],
        input_dims = nLund_variables,
        add_fractions_to_lund=four_vectors)
    model.load_state_dict(dict["model_state_dict"])

    # Load normalizer
    with open("/t3home/rhampp/SVJ_GNN/output/"+name+"/Normalization.pkl",  'rb') as file:
        norm_dict = pickle.load(file)
    means = norm_dict["means"]
    stds = norm_dict["stds"]
    normalizer =  OnTheFlyNormalizer(["x"], means, stds)

    # Put Model into evaluation mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
  
    # Input Data
    datasets = [
        # [base_directory_QCD, f"QCD_{model_type}_eventidx"],

        [base_directory_SVJ + "SVJ_mMed-700GeV_mDark-20GeV_rinv-0.3_alpha-peak", f"SVJ_700_0p3_{model_type}_eventidx"],
        [base_directory_SVJ + "SVJ_mMed-700GeV_mDark-20GeV_rinv-0.5_alpha-peak", f"SVJ_700_0p5_{model_type}_eventidx"],
        [base_directory_SVJ + "SVJ_mMed-700GeV_mDark-20GeV_rinv-0.7_alpha-peak", f"SVJ_700_0p7_{model_type}_eventidx"],

        [base_directory_SVJ + "SVJ_mMed-800GeV_mDark-20GeV_rinv-0.3_alpha-peak", f"SVJ_800_0p3_{model_type}_eventidx"],
        [base_directory_SVJ + "SVJ_mMed-800GeV_mDark-20GeV_rinv-0.5_alpha-peak", f"SVJ_800_0p5_{model_type}_eventidx"],
        [base_directory_SVJ + "SVJ_mMed-800GeV_mDark-20GeV_rinv-0.7_alpha-peak", f"SVJ_800_0p7_{model_type}_eventidx"],

        [base_directory_SVJ + "SVJ_mMed-900GeV_mDark-20GeV_rinv-0.3_alpha-peak", f"SVJ_900_0p3_{model_type}_eventidx"],
        [base_directory_SVJ + "SVJ_mMed-900GeV_mDark-20GeV_rinv-0.5_alpha-peak", f"SVJ_900_0p5_{model_type}_eventidx"],
        [base_directory_SVJ + "SVJ_mMed-900GeV_mDark-20GeV_rinv-0.7_alpha-peak", f"SVJ_900_0p7_{model_type}_eventidx"],

        [base_directory_SVJ + "SVJ_mMed-1000GeV_mDark-20GeV_rinv-0.3_alpha-peak", f"SVJ_1000_0p3_{model_type}_eventidx"],
        [base_directory_SVJ + "SVJ_mMed-1000GeV_mDark-20GeV_rinv-0.5_alpha-peak", f"SVJ_1000_0p5_{model_type}_eventidx"],
        [base_directory_SVJ + "SVJ_mMed-1000GeV_mDark-20GeV_rinv-0.7_alpha-peak", f"SVJ_1000_0p7_{model_type}_eventidx"],

        [base_directory_SVJ + "SVJ_mMed-1100GeV_mDark-20GeV_rinv-0.3_alpha-peak", f"SVJ_1100_0p3_{model_type}_eventidx"],
        [base_directory_SVJ + "SVJ_mMed-1100GeV_mDark-20GeV_rinv-0.5_alpha-peak", f"SVJ_1100_0p5_{model_type}_eventidx"],
        [base_directory_SVJ + "SVJ_mMed-1100GeV_mDark-20GeV_rinv-0.7_alpha-peak", f"SVJ_1100_0p7_{model_type}_eventidx"],

        # [base_directory_SVJ + "SVJ_mMed-1200GeV_mDark-20GeV_rinv-0.3_alpha-peak", f"SVJ_1200_0p3_{model_type}_eventidx"],
        # [base_directory_SVJ + "SVJ_mMed-1200GeV_mDark-20GeV_rinv-0.5_alpha-peak", f"SVJ_1200_0p5_{model_type}_eventidx"],
        # [base_directory_SVJ + "SVJ_mMed-1200GeV_mDark-20GeV_rinv-0.7_alpha-peak", f"SVJ_1200_0p7_{model_type}_eventidx"],

        # [base_directory_SVJ + "SVJ_mMed-1300GeV_mDark-20GeV_rinv-0.3_alpha-peak", f"SVJ_1300_0p3_{model_type}_eventidx"],
        # [base_directory_SVJ + "SVJ_mMed-1300GeV_mDark-20GeV_rinv-0.5_alpha-peak", f"SVJ_1300_0p5_{model_type}_eventidx"],
        # [base_directory_SVJ + "SVJ_mMed-1300GeV_mDark-20GeV_rinv-0.7_alpha-peak", f"SVJ_1300_0p7_{model_type}_eventidx"],

        # [base_directory_SVJ + "SVJ_mMed-1400GeV_mDark-20GeV_rinv-0.3_alpha-peak", f"SVJ_1400_0p3_{model_type}_eventidx"],
        # [base_directory_SVJ + "SVJ_mMed-1400GeV_mDark-20GeV_rinv-0.5_alpha-peak", f"SVJ_1400_0p5_{model_type}_eventidx"],
        # [base_directory_SVJ + "SVJ_mMed-1400GeV_mDark-20GeV_rinv-0.7_alpha-peak", f"SVJ_1400_0p7_{model_type}_eventidx"],

        # [base_directory_SVJ + "SVJ_mMed-1500GeV_mDark-20GeV_rinv-0.3_alpha-peak", f"SVJ_1500_0p3_{model_type}_eventidx"],
        # [base_directory_SVJ + "SVJ_mMed-1500GeV_mDark-20GeV_rinv-0.5_alpha-peak", f"SVJ_1500_0p5_{model_type}_eventidx"],
        # [base_directory_SVJ + "SVJ_mMed-1500GeV_mDark-20GeV_rinv-0.7_alpha-peak", f"SVJ_1500_0p7_{model_type}_eventidx"],

        # [base_directory_SVJ + "SVJ_mMed-3000GeV_mDark-20GeV_rinv-0.3_alpha-peak", f"SVJ_3000_0p3_{model_type}_eventidx"],
        # [base_directory_SVJ + "SVJ_mMed-3000GeV_mDark-20GeV_rinv-0.5_alpha-peak", f"SVJ_3000_0p5_{model_type}_eventidx"],
        # [base_directory_SVJ + "SVJ_mMed-3000GeV_mDark-20GeV_rinv-0.7_alpha-peak", f"SVJ_3000_0p7_{model_type}_eventidx"],

        ]

    model_names = [  
        # "QCD",

        "SVJ_mMed-700GeV_mDark-20GeV_rinv-0.3_alpha-peak/",
        "SVJ_mMed-700GeV_mDark-20GeV_rinv-0.5_alpha-peak/",
        "SVJ_mMed-700GeV_mDark-20GeV_rinv-0.7_alpha-peak/",

        "SVJ_mMed-800GeV_mDark-20GeV_rinv-0.3_alpha-peak/",
        "SVJ_mMed-800GeV_mDark-20GeV_rinv-0.5_alpha-peak/",
        "SVJ_mMed-800GeV_mDark-20GeV_rinv-0.7_alpha-peak/",

        "SVJ_mMed-900GeV_mDark-20GeV_rinv-0.3_alpha-peak/",
        "SVJ_mMed-900GeV_mDark-20GeV_rinv-0.5_alpha-peak/",
        "SVJ_mMed-900GeV_mDark-20GeV_rinv-0.7_alpha-peak/",

        "SVJ_mMed-1000GeV_mDark-20GeV_rinv-0.3_alpha-peak/",
        "SVJ_mMed-1000GeV_mDark-20GeV_rinv-0.5_alpha-peak/",
        "SVJ_mMed-1000GeV_mDark-20GeV_rinv-0.7_alpha-peak/",

        "SVJ_mMed-1100GeV_mDark-20GeV_rinv-0.3_alpha-peak/",
        "SVJ_mMed-1100GeV_mDark-20GeV_rinv-0.5_alpha-peak/",
        "SVJ_mMed-1100GeV_mDark-20GeV_rinv-0.7_alpha-peak/",

        # "SVJ_mMed-1200GeV_mDark-20GeV_rinv-0.3_alpha-peak/",
        # "SVJ_mMed-1200GeV_mDark-20GeV_rinv-0.5_alpha-peak/",
        # "SVJ_mMed-1200GeV_mDark-20GeV_rinv-0.7_alpha-peak/",

        # "SVJ_mMed-1300GeV_mDark-20GeV_rinv-0.3_alpha-peak/",
        # "SVJ_mMed-1300GeV_mDark-20GeV_rinv-0.5_alpha-peak/",
        # "SVJ_mMed-1300GeV_mDark-20GeV_rinv-0.7_alpha-peak/",

        # "SVJ_mMed-1400GeV_mDark-20GeV_rinv-0.3_alpha-peak/",
        # "SVJ_mMed-1400GeV_mDark-20GeV_rinv-0.5_alpha-peak/",
        # "SVJ_mMed-1400GeV_mDark-20GeV_rinv-0.7_alpha-peak/",

        # "SVJ_mMed-1500GeV_mDark-20GeV_rinv-0.3_alpha-peak/",
        # "SVJ_mMed-1500GeV_mDark-20GeV_rinv-0.5_alpha-peak/",
        # "SVJ_mMed-1500GeV_mDark-20GeV_rinv-0.7_alpha-peak/",

        # "SVJ_mMed-3000GeV_mDark-20GeV_rinv-0.3_alpha-peak/",
        # "SVJ_mMed-3000GeV_mDark-20GeV_rinv-0.5_alpha-peak/",
        # "SVJ_mMed-3000GeV_mDark-20GeV_rinv-0.7_alpha-peak/",
                ]


    wp,_ ,_ ,_ ,_ = get_threshold_from_sic(name,file_name='_eval_ROC_AUC.pkl')

    output_directory_tmp = "/work/rhampp/tmp/SVJ_tagger16"
    if not os.path.exists(output_directory_tmp):
        os.mkdir(output_directory_tmp)
        print(f"Created directory: {output_directory_tmp}")

    # Prepare GraphDataLoader
    for model_name, dataset in zip(model_names, datasets):
        output_directory = "root://t3dcachedb03.psi.ch///pnfs/psi.ch/cms/trivcat/store/user/rhampp/Tagged_cut0p85/"


        output_directory = output_directory + model_name

        dataloader = GraphDataLoader_tagger(
            [dataset],
            window_to_load=[[0,1] for _ in range(1)],
            batch_size=1024,
            normalizer=normalizer,)
        
        # Initialize the Tagger dictionary
        tagger = {}

        # Run the graphs through the network and check which one is an SVJ
        start = True
        for graph in dataloader:
            if start:
                tagger["label"] = [graph.y.tolist()]
                tagger["event"] = [graph.event.tolist()]
                graph = graph.to(device)
                out = model(graph)
                score = out[:,0].cpu()

                tagger["score"] = [score.detach().tolist()]
                start = False

            else:
                tagger["label"].append(graph.y.tolist())
                tagger["event"].append(graph.event.tolist())
                graph = graph.to(device)
                out = model(graph)
                score = out[:,0].cpu()

                tagger["score"].append(score.detach().tolist())
        
        tagger["score"] = np.concatenate(tagger["score"])
        tagger["SVJ"] = (np.array(tagger["score"]) > wp)
        tagger["event"] = np.concatenate(tagger["event"])

        # Loop over all files and add an additional branch to indicate whether it is a SVJ or not
        n_tagger = 0
        for file_name in os.listdir(dataset[0]):

            # Check if the file is a ROOT file
            if file_name.endswith(".root"):
                output_file_path = os.path.join(output_directory, file_name)
                output_file_path_tmp = os.path.join(output_directory_tmp, file_name)
                
                # Open the files and save the Events tree.
                with uproot.open(dataset[0] +"/"+ file_name) as infile:
                    tree = infile["Events"]
                    cutflow = infile["CutFlow"].arrays()
                    metadata = infile["Metadata"].arrays()
                    arrays = tree.arrays()
                    n_entries = len(arrays.event)
                    SVJ_tagger_array = [[] for _ in range(n_entries)]

                    # Calulate the truth value for each jet
                    for n in range(n_entries):
                        tag_count = 0
                        tagger_event = []
                           
                        while n_tagger + tag_count < len(tagger["event"]) and tagger["event"][n_tagger + tag_count] == arrays.event[n]:
                            tagger_event.append(tagger["SVJ"][n_tagger + tag_count])
                            tag_count += 1
                        
                        if tag_count > 0:
                            SVJ_tagger_array[n] = tagger_event
                            n_tagger += tag_count

                        else:
                            print("Event numbers don't match:", tagger["event"][n_tagger], arrays.event[n])


                    # Add the new boolean branch to the arrays
                    arrays["FatJet_SVJtagger"] = SVJ_tagger_array

                    trees = {}
                    trees["Events"] = arrays       
                    trees["CutFlow"] = cutflow 
                    trees["Metadata"] = metadata
                

                uproot4Utl.write_root_file(output_file_path_tmp, trees=trees)



                # Copy produced file into storage element
                xrdcp_command = ["xrdcp", output_file_path_tmp, output_file_path]
                try:
                    subprocess.run(xrdcp_command, check=True)
                    print(f"File copied successfully from {output_file_path_tmp} to {output_file_path}")
                except subprocess.CalledProcessError as e:
                    print(f"Error copying file: {e}")

                os.remove(output_file_path_tmp)
            
            # Only do this for the ROOT files in the directory
            else: print(f"{file_name} not a file")


    print(f"==> JetTagger completed in {time.time()-start:.2f} seconds")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--name",
        default="LundNet",
        help="Name of the training to save the losses and the model",
        type=str,
        )

    parser.add_argument(
        "--model",
        default="LN3",
        help="Which model to train: LN3, LN3p, LN3pp, LN5, LN5p, LN5pp (p: save energy fractions, pp:save energyfrac and four vectors)",
        type=str,
        )


    args = parser.parse_args()

    name = args.name
    model_type = args.model
    batch_size = 128

    main(name, model_type)
