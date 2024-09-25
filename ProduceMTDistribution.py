import torch
import ROOT
from architectures.LundNet import LundNet
from utils.LundTreeUtilities import OnTheFlyNormalizer
from data_loading.graphDataLoader_tagger import GraphDataLoader_tagger
from utils.utils_tagJet import get_threshold_from_sic, filter_by_svjtagger

import pickle
import argparse
from utils.utils_training import get_model_parameters

import numpy as np
import os
from collections import defaultdict
import awkward as ak


def main(name, model_type):
    
    # Base directories of the signal and background graphs
    base_directory_SVJ = "/pnfs/psi.ch/cms/trivcat/store/user/rhampp/forTraining/Preselection_applied/SVJ_26_08/"
    base_directory_bkg = "/pnfs/psi.ch/cms/trivcat/store/user/rhampp/forTraining/Preselection_applied/Bkg_sorted/"


    ###------------------------- Load Model and Normalization -------------------------###
    filename = "/" + name + ".pt"
    dict = torch.load("/t3home/rhampp/SVJ_GNN/output/"+ name +"/" + filename,  map_location=torch.device('cpu'))
    nLund_variables, energy_fractions, four_vectors = get_model_parameters(model_type)

    if energy_fractions:
        nLund_variables += 4
 
    # create model and load it
    model = LundNet(
        conv_params = [[128, 128], [128, 128], [64, 64], [64, 64], [32, 32], [32, 32]],
        fc_params = [(32, 0.8)],
        input_dims = nLund_variables,
        add_fractions_to_lund = four_vectors)

    model.load_state_dict(dict["model_state_dict"])

    # Load normalizer
    with open("/t3home/rhampp/SVJ_GNN/output/"+name+"/Normalization.pkl",  'rb') as file:
        norm_dict = pickle.load(file)
    means = norm_dict["means"]
    stds = norm_dict["stds"]
    normalizer =  OnTheFlyNormalizer(["x"], means, stds)

    # Load model on cuda if available and put it in evaluation mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    ###------------------------- Output Name for ROOT Files and Input Dataset Names -------------------------###
    ### ==> Create also Output paths

    output_names = [
        "QCD",
        "TTbar",
        "Wjets",
        "Zjets",
        "SVJ_700_0p3", "SVJ_700_0p5", "SVJ_700_0p7", 
        "SVJ_800_0p3", "SVJ_800_0p5", "SVJ_800_0p7",
        "SVJ_900_0p3", "SVJ_900_0p5", "SVJ_900_0p7",
        "SVJ_1000_0p3", "SVJ_1000_0p5", "SVJ_1000_0p7",
        "SVJ_1100_0p3", "SVJ_1100_0p5", "SVJ_1100_0p7",
        "SVJ_1200_0p3", "SVJ_1200_0p5", "SVJ_1200_0p7",
        "SVJ_1300_0p3", "SVJ_1300_0p5", "SVJ_1300_0p7",
        "SVJ_1400_0p3", "SVJ_1400_0p5", "SVJ_1400_0p7",
        "SVJ_1500_0p3", "SVJ_1500_0p5", "SVJ_1500_0p7",
        "SVJ_3000_0p3", "SVJ_3000_0p5", "SVJ_3000_0p7",
    ]

    all_datasets = [
        [[base_directory_bkg + "QCD/QCD_HT300to500", f"QCD_HT300to500_{model_type}"],
        [base_directory_bkg + "QCD/QCD_HT500to700", f"QCD_HT500to700_{model_type}"],
        [base_directory_bkg + "QCD/QCD_HT700to1000", f"QCD_HT700to1000_{model_type}"],
        [base_directory_bkg + "QCD/QCD_HT1000to1500", f"QCD_HT1000to1500_{model_type}"],
        [base_directory_bkg + "QCD/QCD_HT1500to2000", f"QCD_HT1500to2000_{model_type}"],
        [base_directory_bkg + "QCD/QCD_HT2000toInf", f"QCD_HT2000toInf_{model_type}"],],

        [[base_directory_bkg + "TTJet", f"TTJet_{model_type}"],],

        [[base_directory_bkg + "EW/WJets_inclusive",f"WJets_{model_type}"],],
            
        [[base_directory_bkg + "EW/ZJetsToNuNu_HT-100To200", f"ZJetsToNuNu_HT-100To200_{model_type}"],
        [base_directory_bkg + "EW/ZJetsToNuNu_HT-200To400", f"ZJetsToNuNu_HT-200To400_{model_type}"],
        [base_directory_bkg + "EW/ZJetsToNuNu_HT-400To600", f"ZJetsToNuNu_HT-400To600_{model_type}"],
        [base_directory_bkg + "EW/ZJetsToNuNu_HT-600To800", f"ZJetsToNuNu_HT-600To800_{model_type}"],
        [base_directory_bkg + "EW/ZJetsToNuNu_HT-800To1200", f"ZJetsToNuNu_HT-800To1200_{model_type}"],
        [base_directory_bkg + "EW/ZJetsToNuNu_HT-1200To2500", f"ZJetsToNuNu_HT-1200To2500_{model_type}"],
        [base_directory_bkg + "EW/ZJetsToNuNu_HT-2500ToInf", f"ZJetsToNuNu_HT-2500ToInf_{model_type}"],],
        
        [[base_directory_SVJ + "SVJ_mMed-700GeV_mDark-20GeV_rinv-0.3_alpha-peak", f"SVJ_700_0p3_{model_type}"],],
        [[base_directory_SVJ + "SVJ_mMed-700GeV_mDark-20GeV_rinv-0.5_alpha-peak", f"SVJ_700_0p5_{model_type}"],],
        [[base_directory_SVJ + "SVJ_mMed-700GeV_mDark-20GeV_rinv-0.7_alpha-peak", f"SVJ_700_0p7_{model_type}"],],

        [[base_directory_SVJ + "SVJ_mMed-800GeV_mDark-20GeV_rinv-0.3_alpha-peak", f"SVJ_800_0p3_{model_type}"],],
        [[base_directory_SVJ + "SVJ_mMed-800GeV_mDark-20GeV_rinv-0.5_alpha-peak", f"SVJ_800_0p5_{model_type}"],],
        [[base_directory_SVJ + "SVJ_mMed-800GeV_mDark-20GeV_rinv-0.7_alpha-peak", f"SVJ_800_0p7_{model_type}"],],

        [[base_directory_SVJ + "SVJ_mMed-900GeV_mDark-20GeV_rinv-0.3_alpha-peak", f"SVJ_900_0p3_{model_type}"],],
        [[base_directory_SVJ + "SVJ_mMed-900GeV_mDark-20GeV_rinv-0.5_alpha-peak", f"SVJ_900_0p5_{model_type}"],],
        [[base_directory_SVJ + "SVJ_mMed-900GeV_mDark-20GeV_rinv-0.7_alpha-peak", f"SVJ_900_0p7_{model_type}"],],

        [[base_directory_SVJ + "SVJ_mMed-1000GeV_mDark-20GeV_rinv-0.3_alpha-peak", f"SVJ_1000_0p3_{model_type}"],],
        [[base_directory_SVJ + "SVJ_mMed-1000GeV_mDark-20GeV_rinv-0.5_alpha-peak", f"SVJ_1000_0p5_{model_type}"],],
        [[base_directory_SVJ + "SVJ_mMed-1000GeV_mDark-20GeV_rinv-0.7_alpha-peak", f"SVJ_1000_0p7_{model_type}"],],

        [[base_directory_SVJ + "SVJ_mMed-1100GeV_mDark-20GeV_rinv-0.3_alpha-peak", f"SVJ_1100_0p3_{model_type}"],],
        [[base_directory_SVJ + "SVJ_mMed-1100GeV_mDark-20GeV_rinv-0.5_alpha-peak", f"SVJ_1100_0p5_{model_type}"],],
        [[base_directory_SVJ + "SVJ_mMed-1100GeV_mDark-20GeV_rinv-0.7_alpha-peak", f"SVJ_1100_0p7_{model_type}"],],

        [[base_directory_SVJ + "SVJ_mMed-1200GeV_mDark-20GeV_rinv-0.3_alpha-peak", f"SVJ_1200_0p3_{model_type}"],],
        [[base_directory_SVJ + "SVJ_mMed-1200GeV_mDark-20GeV_rinv-0.5_alpha-peak", f"SVJ_1200_0p5_{model_type}"],],
        [[base_directory_SVJ + "SVJ_mMed-1200GeV_mDark-20GeV_rinv-0.7_alpha-peak", f"SVJ_1200_0p7_{model_type}"],],

        [[base_directory_SVJ + "SVJ_mMed-1300GeV_mDark-20GeV_rinv-0.3_alpha-peak", f"SVJ_1300_0p3_{model_type}"],],
        [[base_directory_SVJ + "SVJ_mMed-1300GeV_mDark-20GeV_rinv-0.5_alpha-peak", f"SVJ_1300_0p5_{model_type}"],],
        [[base_directory_SVJ + "SVJ_mMed-1300GeV_mDark-20GeV_rinv-0.7_alpha-peak", f"SVJ_1300_0p7_{model_type}"],],

        [[base_directory_SVJ + "SVJ_mMed-1400GeV_mDark-20GeV_rinv-0.3_alpha-peak", f"SVJ_1400_0p3_{model_type}"],],
        [[base_directory_SVJ + "SVJ_mMed-1400GeV_mDark-20GeV_rinv-0.5_alpha-peak", f"SVJ_1400_0p5_{model_type}"],],
        [[base_directory_SVJ + "SVJ_mMed-1400GeV_mDark-20GeV_rinv-0.7_alpha-peak", f"SVJ_1400_0p7_{model_type}"],],

        [[base_directory_SVJ + "SVJ_mMed-1500GeV_mDark-20GeV_rinv-0.3_alpha-peak", f"SVJ_1500_0p3_{model_type}"],],
        [[base_directory_SVJ + "SVJ_mMed-1500GeV_mDark-20GeV_rinv-0.5_alpha-peak", f"SVJ_1500_0p5_{model_type}"],],
        [[base_directory_SVJ + "SVJ_mMed-1500GeV_mDark-20GeV_rinv-0.7_alpha-peak", f"SVJ_1500_0p7_{model_type}"],],

        [[base_directory_SVJ + "SVJ_mMed-3000GeV_mDark-20GeV_rinv-0.3_alpha-peak", f"SVJ_3000_0p3_{model_type}"],],
        [[base_directory_SVJ + "SVJ_mMed-3000GeV_mDark-20GeV_rinv-0.5_alpha-peak", f"SVJ_3000_0p5_{model_type}"],],
        [[base_directory_SVJ + "SVJ_mMed-3000GeV_mDark-20GeV_rinv-0.7_alpha-peak", f"SVJ_3000_0p7_{model_type}"],],
     ]
    

    ###------------------------- Define Working Point -------------------------###
    ### Three options: By hand, on basis of the SIC on event level, on the basis of the SIC on jet level

    ## SIC event level
    # wp, _, _, _, _, _, _= get_threshold_from_sic(name,file_name='Eventlevel_ROC.pkl', max_confined=True)

    ## SIC jet level
    # wp, _, _, _, _, _, _= get_threshold_from_sic(name,file_name='_eval_ROC_AUC.pkl', max_confined=True)

    ## By Hand
    wp = 0.5
    
    print("Threshold:", wp)

    # Create output paths
    output_path = os.path.join("/t3home/rhampp/SVJ_GNN/MtDistributions/", name )
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    output_path = os.path.join(output_path, "wp-{:.2f}_newbinning3".format(wp))
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    output_path_0SVJ = os.path.join(output_path, "0SVJ")
    if not os.path.exists(output_path_0SVJ):
        os.mkdir(output_path_0SVJ)

    output_path_1SVJ = os.path.join(output_path, "1SVJ")
    if not os.path.exists(output_path_1SVJ):
        os.mkdir(output_path_1SVJ)
    
    output_path_2SVJ = os.path.join(output_path, "2SVJ")
    if not os.path.exists(output_path_2SVJ):
        os.mkdir(output_path_2SVJ)


    ###------------------------- JetTagger -------------------------###

    ### ==> Step 1: Run datasets through Network ( for each Bkg and signal hypothesis) and safe the score
    for output_name, datasets in zip(output_names,all_datasets):
        print(output_name)
        tagger = {}
        for dataset in datasets:
            print(dataset)

            tagger[dataset[1]] = {}
            score = []
            event = []

            weight = []
            mt = []

            dataloader = GraphDataLoader_tagger(
                [dataset],
                window_to_load=[[0.7,1] for _ in range(1)],
                batch_size=512,
                normalizer=normalizer,)
            
            # Run the graphs through the network and check which one is an SVJ
            for graph in dataloader:
                weight.append(graph.w.tolist())
                mt.append(graph.MT01FatJetMET)
                event.append(graph.event.tolist())
                graph = graph.to(device)
                out = model(graph)
                score_ = out[:,0].cpu()

                score.append(score_.detach().tolist())
            
            score = np.array(np.concatenate(score))
            event = np.array(np.concatenate(event))
            weight = np.array(np.concatenate(weight))
            mt = np.array(np.concatenate(mt))
            
            tagger[dataset[1]]["MT01FatJetMET"] = mt
            tagger[dataset[1]]["SVJ"] = score > wp
            tagger[dataset[1]]["event"] = event
            tagger[dataset[1]]["weight"] = weight

    ### ==> Step 2: Create a dict with 2D arrays with the scores sorted by events
        sorted_data = {}
        sorted_weights = {}

        for key_tagger in tagger:
                
            events = tagger[key_tagger]["event"]
            scores = tagger[key_tagger]["SVJ"]
            weights = tagger[key_tagger]["weight"]
            mts = tagger[key_tagger]["MT01FatJetMET"]

            if events[0] == events[1]:
                start = 0
            else: start = 1

            sorted_data[key_tagger] = {}
            sorted_weights[key_tagger] = []
            sorted_data[key_tagger]["SVJ"] = []
            sorted_data[key_tagger]["MT01FatJetMET"] = []

            for n in np.arange(start,len(events), 2):
                if events[n] == events[n+1]:
                    sorted_data[key_tagger]["MT01FatJetMET"].append([mts[n], mts[n+1]])
                    sorted_data[key_tagger]["SVJ"].append([scores[n], scores[n+1]])
                    sorted_weights[key_tagger].append([weights[n], weights[n+1]])
                else:
                    print("No Matching Events")
    

    ### ==> Step 3: Sort them to the categories 0SVJ, 1SVJ, 2SVJ and create ROOT Files
    
        cat_0SVJ, cat_1SVJ, cat_2SVJ, weights_0SVJ, weights_1SVJ, weights_2SVJ = filter_by_svjtagger(sorted_data, sorted_weights)
         
        lumi = 117100.0

        cat_0SVJ_flat = np.concatenate([ak.flatten(cat_0SVJ[key]["MT01FatJetMET"][:,0], axis=-1) for key in cat_0SVJ])
        weights_0_SVJ_flat = np.concatenate([ak.flatten(weights_0SVJ[key][:,0], axis=-1)for key in weights_0SVJ])

        cat_1SVJ_flat = np.concatenate([ak.flatten(cat_1SVJ[key]["MT01FatJetMET"][:,0], axis=-1) for key in cat_1SVJ])
        weights_1_SVJ_flat = np.concatenate([ak.flatten(weights_1SVJ[key][:,0], axis=-1)for key in weights_1SVJ])

        cat_2SVJ_flat = np.concatenate([ak.flatten(cat_2SVJ[key]["MT01FatJetMET"][:,0], axis=-1) for key in cat_2SVJ])
        weights_2_SVJ_flat = np.concatenate([ak.flatten(weights_2SVJ[key][:,0], axis=-1)for key in weights_2SVJ])
      
        ## Create the Histograms
        # 2 SVJ category has a variable bin width!

        # Ensure proper handling of the weights
        ROOT.TH1F.SetDefaultSumw2()
        ROOT.TH1F.StatOverflows(True)
        h0 = ROOT.TH1F(f"{output_name}" + "_0SVJ" ,"MT01FatJetMET", 50, 650, 2000)

        # Ensure proper handling of the weights
        ROOT.TH1F.SetDefaultSumw2()
        ROOT.TH1F.StatOverflows(True)
        h1 = ROOT.TH1F(f"{output_name}" + "_1SVJ","MT01FatJetMET", 30, 650, 2000)

        # variable_bins_2SVJ = np.array([650. ,750. ,850. ,1000. ,1200. ,1400. ,2000. ], dtype = float)
        # variable_bins_2SVJ = np.array([650, 750, 850, 1000, 1150, 1300, 1450, 1600, 1750, 1900, 2000], dtype = float)
        # variable_bins_2SVJ = np.array([650, 750, 850, 1000, 1150, 1450, 2000], dtype = float)


        # Ensure proper handling of the weights
        ROOT.TH1F.SetDefaultSumw2()
        ROOT.TH1F.StatOverflows(True)
        # h2 = ROOT.TH1F(f"{output_name}" + "_2SVJ","MT01FatJetMET", len(variable_bins_2SVJ)-1, variable_bins_2SVJ)
        h2 = ROOT.TH1F(f"{output_name}" + "_2SVJ","MT01FatJetMET", 10, 650, 2000)

        if len(cat_0SVJ_flat) > 0:
            h0.FillN(len(cat_0SVJ_flat), np.array(cat_0SVJ_flat), np.array(weights_0_SVJ_flat))
        if len(cat_1SVJ_flat) > 0:
            h1.FillN(len(cat_1SVJ_flat), np.array(cat_1SVJ_flat), np.array(weights_1_SVJ_flat))
        if len(cat_2SVJ_flat) > 0:
            h2.FillN(len(cat_2SVJ_flat), np.array(cat_2SVJ_flat), np.array(weights_2_SVJ_flat))


        # Write each category to a root file
        output_file_0SVJ = ROOT.TFile(output_path_0SVJ + "/" + output_name + ".root", "RECREATE")
        h0.Write()
        output_file_0SVJ.Close()

        output_file_1SVJ = ROOT.TFile(output_path_1SVJ + "/" + output_name + ".root", "RECREATE")
        h1.Write()
        output_file_1SVJ.Close()

        output_file_2SVJ = ROOT.TFile(output_path_2SVJ + "/" + output_name + ".root", "RECREATE")
        h2.Write()
        output_file_2SVJ.Close()

        print(f"Created Histograms for {output_name}")
    

    print("==> Created all Histograms")



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
