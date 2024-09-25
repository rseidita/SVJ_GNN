from architectures.LundNet import LundNet
from utils.LundTreeUtilities import OnTheFlyNormalizer
from data_loading.graphDataLoader_tagger import GraphDataLoader_tagger
from utils.utils_training import get_model_parameters
from utils.utils_training import output_directory

from utils.utils_tagJet import roc_event_level, get_threshold_from_sic
import utils.utils_evaluation as ueval

import torch
import pickle
import numpy as np
import os
import awkward as ak
import argparse



def main(name, model_type):
    """ Calculate SIC at Event Level and Plot it
    Args:
        name(str): Name of the saved network 
        model_type(str): What kind of model the network is

    Return:

    """

    # Path to trained model
    filename = "/" + name + ".pt"
    dict = torch.load("/t3home/rhampp/SVJ_GNN/output/"+ name +"/" + filename,  map_location=torch.device('cpu'))
    
    # Load model parameters depending on the LundNet
    nLund_variables, energy_fractions, four_vectors = get_model_parameters(model_type)
    if energy_fractions:
        nLund_variables += 4

    # Create model and load it with the trained model
    model = LundNet(
        conv_params = [[128, 128], [128, 128], [64, 64], [64, 64], [32, 32], [32, 32]],
        fc_params = [(32, 0.8)],
        input_dims = nLund_variables,
        add_fractions_to_lund=four_vectors)

    model.load_state_dict(dict["model_state_dict"])

    # Load normalizer used for the trainnig
    with open("/t3home/rhampp/SVJ_GNN/output/"+name+"/Normalization.pkl",  'rb') as file:
        norm_dict = pickle.load(file)
    means = norm_dict["means"]
    stds = norm_dict["stds"]
    normalizer =  OnTheFlyNormalizer(["x"], means, stds)

    # Load model to cuda if available and switch to evaluation mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()


    # Base path to signal and background directories
    base_directory_SVJ = "/pnfs/psi.ch/cms/trivcat/store/user/rhampp/forTraining/Preselection_applied/SVJ_26_08/"
    base_directory_bkg = "/pnfs/psi.ch/cms/trivcat/store/user/rhampp/forTraining/Preselection_applied/Bkg_sorted/"

    # List: [paths to the "processed" directory containting the .pt files with the graph, name of the .pt file]
    datasets_bkg = [
        [base_directory_bkg + "QCD/QCD_HT300to500", f"QCD_HT300to500_{model_type}"],
        [base_directory_bkg + "QCD/QCD_HT500to700", f"QCD_HT500to700_{model_type}"],
        [base_directory_bkg + "QCD/QCD_HT700to1000", f"QCD_HT700to1000_{model_type}"],
        [base_directory_bkg + "QCD/QCD_HT1000to1500", f"QCD_HT1000to1500_{model_type}"],
        [base_directory_bkg + "QCD/QCD_HT1500to2000", f"QCD_HT1500to2000_{model_type}"],
        [base_directory_bkg + "QCD/QCD_HT2000toInf", f"QCD_HT2000toInf_{model_type}"],

        [base_directory_bkg + "TTJet", f"TTJet_{model_type}"],

        [base_directory_bkg + "EW/WJets_inclusive",f"WJets_{model_type}"],
            
        [base_directory_bkg + "EW/ZJetsToNuNu_HT-100To200", f"ZJetsToNuNu_HT-100To200_{model_type}"],
        [base_directory_bkg + "EW/ZJetsToNuNu_HT-200To400", f"ZJetsToNuNu_HT-200To400_{model_type}"],
        [base_directory_bkg + "EW/ZJetsToNuNu_HT-400To600", f"ZJetsToNuNu_HT-400To600_{model_type}"],
        [base_directory_bkg + "EW/ZJetsToNuNu_HT-600To800", f"ZJetsToNuNu_HT-600To800_{model_type}"],
        [base_directory_bkg + "EW/ZJetsToNuNu_HT-800To1200", f"ZJetsToNuNu_HT-800To1200_{model_type}"],
        [base_directory_bkg + "EW/ZJetsToNuNu_HT-1200To2500", f"ZJetsToNuNu_HT-1200To2500_{model_type}"],
        [base_directory_bkg + "EW/ZJetsToNuNu_HT-2500ToInf", f"ZJetsToNuNu_HT-2500ToInf_{model_type}"],
        ]

    datasets_signal = [
        [base_directory_SVJ + "SVJ_mMed-700GeV_mDark-20GeV_rinv-0.3_alpha-peak", f"SVJ_700_0p3_{model_type}"],
        [base_directory_SVJ + "SVJ_mMed-700GeV_mDark-20GeV_rinv-0.5_alpha-peak", f"SVJ_700_0p5_{model_type}"],
        [base_directory_SVJ + "SVJ_mMed-700GeV_mDark-20GeV_rinv-0.7_alpha-peak", f"SVJ_700_0p7_{model_type}"],

        [base_directory_SVJ + "SVJ_mMed-800GeV_mDark-20GeV_rinv-0.3_alpha-peak", f"SVJ_800_0p3_{model_type}"],
        [base_directory_SVJ + "SVJ_mMed-800GeV_mDark-20GeV_rinv-0.5_alpha-peak", f"SVJ_800_0p5_{model_type}"],
        [base_directory_SVJ + "SVJ_mMed-800GeV_mDark-20GeV_rinv-0.7_alpha-peak", f"SVJ_800_0p7_{model_type}"],

        [base_directory_SVJ + "SVJ_mMed-900GeV_mDark-20GeV_rinv-0.3_alpha-peak", f"SVJ_900_0p3_{model_type}"],
        [base_directory_SVJ + "SVJ_mMed-900GeV_mDark-20GeV_rinv-0.5_alpha-peak", f"SVJ_900_0p5_{model_type}"],
        [base_directory_SVJ + "SVJ_mMed-900GeV_mDark-20GeV_rinv-0.7_alpha-peak", f"SVJ_900_0p7_{model_type}"],

        [base_directory_SVJ + "SVJ_mMed-1000GeV_mDark-20GeV_rinv-0.3_alpha-peak", f"SVJ_1000_0p3_{model_type}"],
        [base_directory_SVJ + "SVJ_mMed-1000GeV_mDark-20GeV_rinv-0.5_alpha-peak", f"SVJ_1000_0p5_{model_type}"],
        [base_directory_SVJ + "SVJ_mMed-1000GeV_mDark-20GeV_rinv-0.7_alpha-peak", f"SVJ_1000_0p7_{model_type}"],

        [base_directory_SVJ + "SVJ_mMed-1100GeV_mDark-20GeV_rinv-0.3_alpha-peak", f"SVJ_1100_0p3_{model_type}"],
        [base_directory_SVJ + "SVJ_mMed-1100GeV_mDark-20GeV_rinv-0.5_alpha-peak", f"SVJ_1100_0p5_{model_type}"],
        [base_directory_SVJ + "SVJ_mMed-1100GeV_mDark-20GeV_rinv-0.7_alpha-peak", f"SVJ_1100_0p7_{model_type}"],

        [base_directory_SVJ + "SVJ_mMed-1200GeV_mDark-20GeV_rinv-0.3_alpha-peak", f"SVJ_1200_0p3_{model_type}"],
        [base_directory_SVJ + "SVJ_mMed-1200GeV_mDark-20GeV_rinv-0.5_alpha-peak", f"SVJ_1200_0p5_{model_type}"],
        [base_directory_SVJ + "SVJ_mMed-1200GeV_mDark-20GeV_rinv-0.7_alpha-peak", f"SVJ_1200_0p7_{model_type}"],

        [base_directory_SVJ + "SVJ_mMed-1300GeV_mDark-20GeV_rinv-0.3_alpha-peak", f"SVJ_1300_0p3_{model_type}"],
        [base_directory_SVJ + "SVJ_mMed-1300GeV_mDark-20GeV_rinv-0.5_alpha-peak", f"SVJ_1300_0p5_{model_type}"],
        [base_directory_SVJ + "SVJ_mMed-1300GeV_mDark-20GeV_rinv-0.7_alpha-peak", f"SVJ_1300_0p7_{model_type}"],

        [base_directory_SVJ + "SVJ_mMed-1400GeV_mDark-20GeV_rinv-0.3_alpha-peak", f"SVJ_1400_0p3_{model_type}"],
        [base_directory_SVJ + "SVJ_mMed-1400GeV_mDark-20GeV_rinv-0.5_alpha-peak", f"SVJ_1400_0p5_{model_type}"],
        [base_directory_SVJ + "SVJ_mMed-1400GeV_mDark-20GeV_rinv-0.7_alpha-peak", f"SVJ_1400_0p7_{model_type}"],

        [base_directory_SVJ + "SVJ_mMed-1500GeV_mDark-20GeV_rinv-0.3_alpha-peak", f"SVJ_1500_0p3_{model_type}"],
        [base_directory_SVJ + "SVJ_mMed-1500GeV_mDark-20GeV_rinv-0.5_alpha-peak", f"SVJ_1500_0p5_{model_type}"],
        [base_directory_SVJ + "SVJ_mMed-1500GeV_mDark-20GeV_rinv-0.7_alpha-peak", f"SVJ_1500_0p7_{model_type}"],

        # [base_directory_SVJ + "SVJ_mMed-3000GeV_mDark-20GeV_rinv-0.3_alpha-peak", f"SVJ_3000_0p3_{model_type}"],
        # [base_directory_SVJ + "SVJ_mMed-3000GeV_mDark-20GeV_rinv-0.5_alpha-peak", f"SVJ_3000_0p5_{model_type}"],
        # [base_directory_SVJ + "SVJ_mMed-3000GeV_mDark-20GeV_rinv-0.7_alpha-peak", f"SVJ_3000_0p7_{model_type}"],
    ]

    # Create names for the ROC dictionary
    model_names = [
        "SVJ_700_0p3", "SVJ_700_0p5", "SVJ_700_0p7", 
        "SVJ_800_0p3", "SVJ_800_0p5", "SVJ_800_0p7",
        "SVJ_900_0p3", "SVJ_900_0p5", "SVJ_900_0p7",
        "SVJ_1000_0p3", "SVJ_1000_0p5", "SVJ_1000_0p7",
        "SVJ_1100_0p3", "SVJ_1100_0p5", "SVJ_1100_0p7",
        "SVJ_1200_0p3", "SVJ_1200_0p5", "SVJ_1200_0p7",
        "SVJ_1300_0p3", "SVJ_1300_0p5", "SVJ_1300_0p7",
        "SVJ_1400_0p3", "SVJ_1400_0p5", "SVJ_1400_0p7",
        "SVJ_1500_0p3", "SVJ_1500_0p5", "SVJ_1500_0p7",
        # "SVJ_3000_0p3", "SVJ_3000_0p5", "SVJ_3000_0p7",
    ]
    
    ###--------- Run BKG through model -----------------------------------------------------------
    ##### Use a GraphDataLoader without random weighted sampling and weight each jet afterwards
    tagger_bkg = {}
    GNN_score_bkg = []
    GNN_weight_bkg = []
    dataloader_bkg = GraphDataLoader_tagger(
            datasets_bkg,
            window_to_load=[[0.7,1] for _ in range(len(datasets_bkg))],
            batch_size=512,
            normalizer=normalizer,
        )

    ## Run the graphs through the network in batches
    start = True
    for graph in dataloader_bkg:
        if start:
            tagger_bkg["label"] = [graph.y.tolist()]
            tagger_bkg["event"] = [graph.event.tolist()]
            tagger_bkg["weight"] = [graph.w.tolist()]
            graph = graph.to(device)
            out = model(graph)
            score = out[:,0].cpu()

            tagger_bkg["score"] = [score.detach().tolist()]
            start = False

        else:
            tagger_bkg["label"].append(graph.y.tolist())
            tagger_bkg["event"].append(graph.event.tolist())
            tagger_bkg["weight"].append(graph.w.tolist())
            graph = graph.to(device)
            out = model(graph)
            score = out[:,0].cpu()

            tagger_bkg["score"].append(score.detach().tolist())

    # Concatenate the output
    tagger_bkg["score"] = np.concatenate(tagger_bkg["score"])
    tagger_bkg["event"] = np.concatenate(tagger_bkg["event"])
    tagger_bkg["weight"] = np.concatenate(tagger_bkg["weight"])

    # Check which jet is a semi-visible jet according to the network
    total_events = len(set(tagger_bkg["event"]))
    n_tagger = 0
    for n_event in range(total_events):
        event_number = tagger_bkg["event"][n_tagger]
        n_jet = 0
        score_event = []
        weight_event = []
        while (n_tagger + n_jet < len(tagger_bkg["event"])) and (tagger_bkg["event"][n_tagger + n_jet] == event_number):
            if n_jet < 2:
                score_event.append(tagger_bkg["score"][n_tagger + n_jet])
                weight_event.append(tagger_bkg["weight"][n_tagger + n_jet])
            n_jet += 1
        n_tagger += n_jet
        GNN_score_bkg.append(score_event)
        GNN_weight_bkg.append(weight_event)


    # Initalize dict to save roc and create the keys
    roc = {}
    model_names = [
        "SVJ_700_0p3", "SVJ_700_0p5", "SVJ_700_0p7", 
        "SVJ_800_0p3", "SVJ_800_0p5", "SVJ_800_0p7",
        "SVJ_900_0p3", "SVJ_900_0p5", "SVJ_900_0p7",
        "SVJ_1000_0p3", "SVJ_1000_0p5", "SVJ_1000_0p7",
        "SVJ_1100_0p3", "SVJ_1100_0p5", "SVJ_1100_0p7",
        "SVJ_1200_0p3", "SVJ_1200_0p5", "SVJ_1200_0p7",
        "SVJ_1300_0p3", "SVJ_1300_0p5", "SVJ_1300_0p7",
        "SVJ_1400_0p3", "SVJ_1400_0p5", "SVJ_1400_0p7",
        "SVJ_1500_0p3", "SVJ_1500_0p5", "SVJ_1500_0p7",
        # "SVJ_3000_0p3", "SVJ_3000_0p5", "SVJ_3000_0p7",
    ]


    ###--------- Run Signal through model -----------------------------------------------------------
    #--------------and find for each signal the maximal SIC--------------------------------------
    for model_name, dataset in zip(model_names, datasets_signal):
        Target_score_signal = []
        GNN_score_signal = []
        GNN_weight_sig = []

        dataloader = GraphDataLoader_tagger(
            [dataset],
            window_to_load=[[0.7,1] for _ in range(1)],
            batch_size=512,
            normalizer=normalizer,
        )

        # Dictionary to fill with the output
        tagger = {}

        ## Run the graphs through the network
        start = True
        for graph in dataloader:
            if start:
                tagger["label"] = [graph.y.tolist()]
                tagger["weight"] = [graph.w.tolist()]
                tagger["event"] = [graph.event.tolist()]
                graph = graph.to(device)
                out = model(graph)
                score = out[:,0].cpu()

                tagger["score"] = [score.detach().tolist()]
                start = False

            else:
                tagger["label"].append(graph.y.tolist())
                tagger["weight"].append(graph.w.tolist())
                tagger["event"].append(graph.event.tolist())
                graph = graph.to(device)
                out = model(graph)
                score = out[:,0].cpu()

                tagger["score"].append(score.detach().tolist())
        
        tagger["score"] = np.concatenate(tagger["score"])
        tagger["event"] = np.concatenate(tagger["event"])
        tagger["weight"] = np.concatenate(tagger["weight"])

        total_events = len(set(tagger["event"]))
        Target_score_signal.append([1]*total_events)

        # Check which jet is a semi-visible jet according to the network
        n_tagger = 0
        for n_event in range(total_events):
            event_number = tagger["event"][n_tagger]
            n_jet = 0
            score_event = []
            weight_event = []
            while (n_tagger+ n_jet < len(tagger["event"])) and (tagger["event"][n_tagger+n_jet] == event_number):
                if n_jet < 2:
                    score_event.append(tagger["score"][n_tagger+n_jet])
                    weight_event.append(tagger["weight"][n_tagger+n_jet])
                n_jet += 1
            n_tagger += n_jet
            GNN_score_signal.append(score_event)
            GNN_weight_sig.append(weight_event)
        

        # Calculate ROC and SIC and save it in a dict
        tpr, fpr, threshold = roc_event_level(GNN_score_signal, GNN_score_bkg, GNN_weight_sig, GNN_weight_bkg)
        sic = np.zeros_like(tpr, dtype=float)
        np.divide(tpr, np.sqrt(fpr), out=sic, where= np.sqrt(fpr)>0)
        
        # Fill the ROC dict
        roc[model_name] = {   
            "fpr": fpr,
            "tpr": tpr,
            "threshold": threshold,
            "sic": sic,}
    

    ###------------- Save ROC to output ----------------------------------------------------
    # output_path = output_directory(name)
    # with open(output_path + "/" + "Eventlevel_ROC.pkl", 'wb') as f:
    #     pickle.dump(roc, f)
    # print(f"Evaluation data saved in {os.path.join(output_path, 'Eventlevel_ROC.pkl')}")


    ###---------------- Plot SIC --------------------------

    ueval.plot_sic(roc,name,suffix = "_EventLevel")
    ueval.plot_sic_small(roc, name, suffix = "_EventLevel")



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

    main(name, model_type)










