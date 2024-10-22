from sklearn.metrics import roc_curve, roc_auc_score
import numpy as np
from data_loading.graphDataLoader import GraphDataLoader
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import re
import pickle
import os
import dcor

from utils.utils_training import output_directory

def Evaluate_model(base_directory_bkg, base_directory_SVJ, model_type, eval_type, batch_size, normalizer, model, device, window_to_load= [0.7,1], jet_idx=None):
    """Evaluate the model on each signal dataset independently

    Args:
        base_directory_QCD(str): Path to the bkg directory
        base_directory_SVJ(str): Path to the signal directroy
        model_type (str): Type of the model: LN3, LN3p, LN5, LN5+
        eval_type (strg): _ + on which graphs to evaluate: "", "_eventidx" (to get mt distribution), "_3J" (to get evaluation on single jet)
        batch_size(int)
        normalizer
        model
        device
        permutation: permutation from the dataloader
        window_to_load: for the evaluation: standard = [0.7,1]
        jet_idx(int): Which jet to look at (0,1,2), default = None -> look at all graphs in the window


    Returns:   evaluation_dict[SVJ_***_*p*] = {
                'weights': np.concatenate(weights),
                'target_labels': np.concatenate(target_labels),
                'scores': np.concatenate(test_scores)}
        
    """


    datasets_signal = [
        [base_directory_SVJ + "SVJ_mMed-700GeV_mDark-20GeV_rinv-0.3_alpha-peak", f"SVJ_700_0p3_{model_type}{eval_type}"],
        [base_directory_SVJ + "SVJ_mMed-700GeV_mDark-20GeV_rinv-0.5_alpha-peak", f"SVJ_700_0p5_{model_type}{eval_type}"],
        [base_directory_SVJ + "SVJ_mMed-700GeV_mDark-20GeV_rinv-0.7_alpha-peak", f"SVJ_700_0p7_{model_type}{eval_type}"],

        [base_directory_SVJ + "SVJ_mMed-800GeV_mDark-20GeV_rinv-0.3_alpha-peak", f"SVJ_800_0p3_{model_type}{eval_type}"],
        [base_directory_SVJ + "SVJ_mMed-800GeV_mDark-20GeV_rinv-0.5_alpha-peak", f"SVJ_800_0p5_{model_type}{eval_type}"],
        [base_directory_SVJ + "SVJ_mMed-800GeV_mDark-20GeV_rinv-0.7_alpha-peak", f"SVJ_800_0p7_{model_type}{eval_type}"],

        [base_directory_SVJ + "SVJ_mMed-900GeV_mDark-20GeV_rinv-0.3_alpha-peak", f"SVJ_900_0p3_{model_type}{eval_type}"],
        [base_directory_SVJ + "SVJ_mMed-900GeV_mDark-20GeV_rinv-0.5_alpha-peak", f"SVJ_900_0p5_{model_type}{eval_type}"],
        [base_directory_SVJ + "SVJ_mMed-900GeV_mDark-20GeV_rinv-0.7_alpha-peak", f"SVJ_900_0p7_{model_type}{eval_type}"],

        [base_directory_SVJ + "SVJ_mMed-1000GeV_mDark-20GeV_rinv-0.3_alpha-peak", f"SVJ_1000_0p3_{model_type}{eval_type}"],
        [base_directory_SVJ + "SVJ_mMed-1000GeV_mDark-20GeV_rinv-0.5_alpha-peak", f"SVJ_1000_0p5_{model_type}{eval_type}"],
        [base_directory_SVJ + "SVJ_mMed-1000GeV_mDark-20GeV_rinv-0.7_alpha-peak", f"SVJ_1000_0p7_{model_type}{eval_type}"],

        [base_directory_SVJ + "SVJ_mMed-1100GeV_mDark-20GeV_rinv-0.3_alpha-peak", f"SVJ_1100_0p3_{model_type}{eval_type}"],
        [base_directory_SVJ + "SVJ_mMed-1100GeV_mDark-20GeV_rinv-0.5_alpha-peak", f"SVJ_1100_0p5_{model_type}{eval_type}"],
        [base_directory_SVJ + "SVJ_mMed-1100GeV_mDark-20GeV_rinv-0.7_alpha-peak", f"SVJ_1100_0p7_{model_type}{eval_type}"],

        [base_directory_SVJ + "SVJ_mMed-1200GeV_mDark-20GeV_rinv-0.3_alpha-peak", f"SVJ_1200_0p3_{model_type}{eval_type}"],
        [base_directory_SVJ + "SVJ_mMed-1200GeV_mDark-20GeV_rinv-0.5_alpha-peak", f"SVJ_1200_0p5_{model_type}{eval_type}"],
        [base_directory_SVJ + "SVJ_mMed-1200GeV_mDark-20GeV_rinv-0.7_alpha-peak", f"SVJ_1200_0p7_{model_type}{eval_type}"],

        [base_directory_SVJ + "SVJ_mMed-1300GeV_mDark-20GeV_rinv-0.3_alpha-peak", f"SVJ_1300_0p3_{model_type}{eval_type}"],
        [base_directory_SVJ + "SVJ_mMed-1300GeV_mDark-20GeV_rinv-0.5_alpha-peak", f"SVJ_1300_0p5_{model_type}{eval_type}"],
        [base_directory_SVJ + "SVJ_mMed-1300GeV_mDark-20GeV_rinv-0.7_alpha-peak", f"SVJ_1300_0p7_{model_type}{eval_type}"],

        [base_directory_SVJ + "SVJ_mMed-1400GeV_mDark-20GeV_rinv-0.3_alpha-peak", f"SVJ_1400_0p3_{model_type}{eval_type}"],
        [base_directory_SVJ + "SVJ_mMed-1400GeV_mDark-20GeV_rinv-0.5_alpha-peak", f"SVJ_1400_0p5_{model_type}{eval_type}"],
        [base_directory_SVJ + "SVJ_mMed-1400GeV_mDark-20GeV_rinv-0.7_alpha-peak", f"SVJ_1400_0p7_{model_type}{eval_type}"],

        [base_directory_SVJ + "SVJ_mMed-1500GeV_mDark-20GeV_rinv-0.3_alpha-peak", f"SVJ_1500_0p3_{model_type}{eval_type}"],
        [base_directory_SVJ + "SVJ_mMed-1500GeV_mDark-20GeV_rinv-0.5_alpha-peak", f"SVJ_1500_0p5_{model_type}{eval_type}"],
        [base_directory_SVJ + "SVJ_mMed-1500GeV_mDark-20GeV_rinv-0.7_alpha-peak", f"SVJ_1500_0p7_{model_type}{eval_type}"],

        [base_directory_SVJ + "SVJ_mMed-3000GeV_mDark-20GeV_rinv-0.3_alpha-peak", f"SVJ_3000_0p3_{model_type}{eval_type}"],
        [base_directory_SVJ + "SVJ_mMed-3000GeV_mDark-20GeV_rinv-0.5_alpha-peak", f"SVJ_3000_0p5_{model_type}{eval_type}"],
        [base_directory_SVJ + "SVJ_mMed-3000GeV_mDark-20GeV_rinv-0.7_alpha-peak", f"SVJ_3000_0p7_{model_type}{eval_type}"],
    ]

    datasets_bkg = [

        [base_directory_bkg + "QCD/QCD_HT300to500", f"QCD_HT300to500_{model_type}{eval_type}"],
        [base_directory_bkg + "QCD/QCD_HT500to700", f"QCD_HT500to700_{model_type}{eval_type}"],
        [base_directory_bkg + "QCD/QCD_HT700to1000", f"QCD_HT700to1000_{model_type}{eval_type}"],
        [base_directory_bkg + "QCD/QCD_HT1000to1500", f"QCD_HT1000to1500_{model_type}{eval_type}"],
        [base_directory_bkg + "QCD/QCD_HT1500to2000", f"QCD_HT1500to2000_{model_type}{eval_type}"],
        [base_directory_bkg + "QCD/QCD_HT2000toInf", f"QCD_HT2000toInf_{model_type}{eval_type}"],

        [base_directory_bkg + "TTJet", f"TTJet_{model_type}{eval_type}"],

        [base_directory_bkg + "EW/WJets_inclusive",f"WJets_{model_type}{eval_type}"],
            
        [base_directory_bkg + "EW/ZJetsToNuNu_HT-100To200", f"ZJetsToNuNu_HT-100To200_{model_type}{eval_type}"],
        [base_directory_bkg + "EW/ZJetsToNuNu_HT-200To400", f"ZJetsToNuNu_HT-200To400_{model_type}{eval_type}"],
        [base_directory_bkg + "EW/ZJetsToNuNu_HT-400To600", f"ZJetsToNuNu_HT-400To600_{model_type}{eval_type}"],
        [base_directory_bkg + "EW/ZJetsToNuNu_HT-600To800", f"ZJetsToNuNu_HT-600To800_{model_type}{eval_type}"],
        [base_directory_bkg + "EW/ZJetsToNuNu_HT-800To1200", f"ZJetsToNuNu_HT-800To1200_{model_type}{eval_type}"],
        [base_directory_bkg + "EW/ZJetsToNuNu_HT-1200To2500", f"ZJetsToNuNu_HT-1200To2500_{model_type}{eval_type}"],
        [base_directory_bkg + "EW/ZJetsToNuNu_HT-2500ToInf", f"ZJetsToNuNu_HT-2500ToInf_{model_type}{eval_type}"],
    ]

    evaluation_dict = {}

    ## Run Bkg through network
    weights_bkg = []
    target_labels_bkg = []
    test_scores_bkg = []
    graph_label_bkg = []
    mt_bkg = []
    event_number_bkg = []

    for dataset in datasets_bkg:
        dataloader_bkg = GraphDataLoader(
            [dataset],
            window_to_load=[window_to_load for _ in range(1)],
            batch_size=batch_size,
            normalizer=normalizer,
            jet_idx=jet_idx,)
        
        model.eval()

        for graphs in dataloader_bkg:
            weights_bkg.append(np.array(graphs.w))
            mt_bkg.append(np.array(graphs.MT01FatJetMET))
            event_number_bkg.append(np.array(graphs.event))
            graphs = graphs.to(device)
            target = graphs.y
            out = model(graphs)
            score = out[:,0].cpu()
            target_labels_bkg.append(np.array(target.cpu()))
            test_scores_bkg.append(score.detach().numpy())
            graph_label_bkg.append([dataset[1]]*len(score.detach().numpy()))
            
        
    weights_bkg =  np.concatenate(weights_bkg)
    mt_bkg = np.concatenate(mt_bkg)
    test_scores_bkg = np.concatenate(test_scores_bkg)
    target_labels_bkg = np.concatenate(target_labels_bkg)
    graph_label_bkg = np.concatenate(graph_label_bkg)   
    event_number_bkg = np.concatenate(event_number_bkg)



    # Run signal through network and safe it togheter with the bkg
    for dataset in datasets_signal:
        weights_sig = []
        target_labels_sig = []
        test_scores_sig = []
        graph_label_sig = []
        mt_sig = []
        event_number_sig = []

        dataloader_sig = GraphDataLoader(
            [dataset],
            window_to_load=[window_to_load for _ in range(1)],
            batch_size=batch_size,
            normalizer=normalizer,
            jet_idx=jet_idx,)
    
        model.eval()

        for graphs in dataloader_sig:
            weights_sig.append(np.array(graphs.w))
            mt_sig.append(np.array(graphs.MT01FatJetMET))
            event_number_sig.append(np.array(graphs.event))
            graphs = graphs.to(device)
            target = graphs.y
            out = model(graphs)
            score = out[:,0].cpu()
            target_labels_sig.append(np.array(target.cpu()))
            test_scores_sig.append(score.detach().numpy())
            graph_label_sig.append([dataset[1]]*len(score.detach().numpy()))

        weights_sig =  np.concatenate(weights_sig)
        mt_sig = np.concatenate(mt_sig)
        test_scores_sig = np.concatenate(test_scores_sig)
        target_labels_sig = np.concatenate(target_labels_sig)
        graph_label_sig = np.concatenate(graph_label_sig)
        event_number_sig = np.concatenate(event_number_sig)

        weights = np.append(weights_sig, weights_bkg)
        mt = np.append(mt_sig, mt_bkg)
        test_scores = np.append(test_scores_sig, test_scores_bkg)
        target_labels = np.append(target_labels_sig, target_labels_bkg)
        graph_label = np.append(graph_label_sig, graph_label_bkg)
        event_number = np.append(event_number_sig, event_number_bkg)

        disco = dcor.distance_correlation(np.array(test_scores), np.array(mt))

        print("DisCo = ", disco)

        evaluation_dict[dataset[1]] = {
            'weights': weights,
            'target_labels': target_labels,
            'scores': test_scores,
            'mt': mt,
            'dataset_labels':graph_label,
            'disco': disco,
            'event': event_number,
        }
    
    
    return evaluation_dict

def roc_auc(evaluation_dict, name, suffix="eval"):
    """ Calculate ROC and AUC

    Args:
        evaluation_dict(dict): with weights, target_labels, scorces

    Returns: roc(dict), auc(dict)
        
    """

    output = {}
    roc = {}
    auc = {}

    for key in evaluation_dict:
        target = evaluation_dict[key]["target_labels"]
        score  = evaluation_dict[key]["scores"]
        weights = evaluation_dict[key]["weights"]
        mt = evaluation_dict[key]["mt"]
        dataset_labels = evaluation_dict[key]["dataset_labels"]
        disco = evaluation_dict[key]["disco"]
        event = evaluation_dict[key]["event"]

        if np.all(target == 0) or np.all(target == 1):
            continue

        # fpr_, tpr_, thres_ = roc_curve(target, score, sample_weight=weights)
        # auc_ = roc_auc_score(target, score, sample_weight=weights)

        # Already weighted by random weighted sampler!
        fpr_, tpr_, thres_ = roc_curve(target, score)
        auc_ = roc_auc_score(target, score)

        #calculate sic
        sic_ = np.zeros_like(tpr_, dtype=float)
        np.divide(tpr_, np.sqrt(fpr_), out=sic_, where=fpr_ != 0)

        output[key]= {
            "fpr": fpr_,
            "tpr": tpr_,
            "threshold": thres_, 
            "auc": auc_,
            "target_labels": target,
            "scores": score,
            "weight": weights,
            "sic": sic_,
            "disco": disco,
            "MT01FatJet": mt,
            "dataset_labels": dataset_labels,
            "event": event
            }
        
        roc[key] = {   
            "fpr": fpr_,
            "tpr": tpr_,
            "threshold": thres_,
            "sic": sic_,
            }
        
        auc[key] = auc_
       

    output_path = output_directory(name)
    with open(output_path + "/" + suffix + "_ROC_AUC.pkl", 'wb') as f:
        pickle.dump(output, f)
    print(f"Evaluation data saved in {os.path.join(output_path, suffix, '_ROC_AUC.pkl')}")

    return roc, auc


def make_dict(auc):
    """Create Dictionary with 

    Args:
        auc(dict): with keys SVJ_***_*p*

    Returns:  auc(dict) with the form: {(rinv, mMed): 0.6106412792103517, ...
        
    """
    dict_result = {}

    for key in auc.keys():
        pattern = r"SVJ_(\d+)_0p(\d+)"
        match = re.search(pattern, key)
        if match:
            mMed = int(match.group(1))
            rinv = float(f"0.{match.group(2)}")
            sig_key = (mMed, rinv)
            dict_result[sig_key] = auc[key]
    return dict_result

def add_legends_low(ax, legends, loc='lower right', spacing=0.26):
    """
    Add multiple legends to a Matplotlib axes.

    Parameters:
        ax: The axes to which legends are added.
        legends: A list of legend handles, each a list of Line2D objects to be included in a legend.
        loc: Location of the legends (default 'upper right').
        spacing: Horizontal space between legends (default 0.1).
    """
    num_legends = len(legends)
    base_anchor = (0.995, 0.005)

    
    for i, handles in enumerate(legends):
        # Adjust the x-coordinate of the bbox_to_anchor for each legend
        x_offset = base_anchor[0] - i * spacing
        legend = ax.legend(handles=handles, loc=loc, bbox_to_anchor=(x_offset, base_anchor[1]))
        ax.add_artist(legend)

def plot_roc(roc, name, suffix = "_eval"):

    # Define line styles for each value of rinv
    line_styles = {
        '0.3': '-',
        '0.5': '--',
        '0.7': ':'
    }

    colors = {
        '700': 'rebeccapurple',        
        '800': 'plum',     
        '900': 'lightcoral',    
        '1000': 'firebrick',        
        '1100': 'gold',      
        '1200': 'olive',     
        '1300': 'yellowgreen',     
        '1400': '#A8DADC',      # powder blue
        '1500': '#457B9D',    # Steel blue
        '3000': '#1D3557',    # Dark blue
    }

    plt.figure()
    ax = plt.gca()
 
    # Iterate over each key in roc dictionary and plot the curves
    for key in roc:
        parts = key.split("_")
        label = f"$M_{{Z'}} = {parts[1]},  r_{{inv}} = 0.{parts[2].split('p')[1]}$"
        rinv_value = f"0.{parts[2].split('p')[1]}"
        m_z = parts[1]
        plt.plot(roc[key]["fpr"], roc[key]["tpr"], color=colors[m_z], linewidth = 0.9, linestyle=line_styles[rinv_value], label=label)

    # Create dummy plots for separate legends
    # Dummy plots for MZ values
    legend_handles_mz = [plt.Line2D([0], [1], color=color, linewidth = 0.9,linestyle='-', label=f"$M_{{Z'}} = {m_z}$") for m_z, color in colors.items()]
    # Dummy plots for rinv values
    legend_handles_rinv = [plt.Line2D([0], [1], color='black',linewidth = 0.9, linestyle=linestyle, label=f"$r_{{inv}} = {rinv}$") for rinv, linestyle in line_styles.items()]

    # Plot the actual ROC curves
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    # Create the first legend for MZ values
    add_legends_low(ax,[legend_handles_mz,legend_handles_rinv])
    plt.grid()

    output_path = output_directory(name)
    outout_plot_name = name + suffix + "_roc.png"
    print(f"{outout_plot_name} has been saved.")
    plt.savefig(os.path.join(output_path,outout_plot_name), dpi=400)
    plt.close()





def plot_auc(auc, name, suffix = "_eval",  variable_name = "AUC"):
    """Plot AUC for all Mediator masses and rinv in one plot

    Args:
        auc(dict): as dict created in roc_auc()
        name(str): Name of the training for the saving of the pdf
        variable_name(str): Default AUC

    Returns:
        
    """

    auc = make_dict(auc)
        
    x_list = []
    y_list = []
    for key in auc.keys():
        x, y = key
        if x not in x_list: x_list.append(x)
        if y not in y_list: y_list.append(y)
    x_list = sorted(x_list)
    y_list = sorted(y_list)[::-1]  # ascending y axis

    gridxy = []
    for idx_y, y in enumerate(y_list):
        gridxy.append([])
        for idx_x, x in enumerate(x_list):
            key = (x, y)
            if key in auc.keys():
                value = auc[key]
            else:
                value = 0.
            gridxy[idx_y].append(value)

    # Create a copy of the 'viridis' colormap
    cmap = plt.get_cmap('viridis')

    # Create a new colormap that is the same as 'viridis', but with the color for the specific values set to white
    new_cmap = colors.LinearSegmentedColormap.from_list(
            'new_cmap', cmap(np.linspace(0, 1, cmap.N)), N=cmap.N)

    # Set the color for the specific values to white
    new_cmap.set_under('white', 0.01)

    # Draw AUC table using matplotlib
    plt.figure(figsize=(12, 12))
    im  = plt.imshow(gridxy, cmap=new_cmap, interpolation='nearest')
    # Show value auc in each bin
    for idx_x, x in enumerate(x_list):
        for idx_y, y in enumerate(y_list):
            plt.text(idx_x, idx_y, "{:.3f}".format(gridxy[idx_y][idx_x]), va='center', ha='center', color='white', fontsize=18)
  
    im.set_clim(0.5, 1.0)
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label(variable_name)
    
    plt.xticks(np.arange(len(x_list)), x_list)
    plt.yticks(np.arange(len(y_list)), y_list)
    plt.xlabel("$r_{inv}$")
    plt.ylabel("$M_{Z'}$")

    #hep.cms.label('Preliminary', data=False)
    #hep.cms.label(loc=0)

    output_path = output_directory(name)
    outout_plot_name = name + suffix + "_auc.png"
    print(f"{outout_plot_name} has been saved.")
    plt.savefig(os.path.join(output_path,outout_plot_name),  dpi=400)
    plt.close()
    
    return 


def plot_scores(evaluation_dict, name, suffix= "_eval"):
    """Plot scores for all Mediator masses and rinv in one plot

    Args:
        evaluation_dict(dict): a dict created in Evaluate_model()
        name(str): Name of the training for the saving of the pdf
       

    Returns:
    """
        

    line_styles = {
        '0.3': '-',
        '0.5': '--',
        '0.7': ':'
    }

    # colors = {
    #     '700': 'plum',        # Plum
    #     '800': '#E63946',     # Firebrick
    #     '900': '#F4A261',     # Tangerine yellow
    #     '1000': '#A8DADC',    # Powder blue
    #     '1500': '#457B9D',    # Steel blue
    #     '3000': '#1D3557',    # Dark blue
    #     'qcd': 'dimgrey',
    # }


    colors = {
        '700': 'rebeccapurple',        
        '800': 'plum',     
        '900': 'lightcoral',    
        '1000': 'firebrick',        
        '1100': 'gold',      
        '1200': 'olive',     
        '1300': 'yellowgreen',     
        '1400': '#A8DADC',      # powder blue
        '1500': '#457B9D',    # Steel blue
        '3000': '#1D3557',    # Dark blue
        'qcd': 'dimgrey',  
    }
 
    # Iterate over each key in roc dictionary and add the scores
    qcd = True

    plt.figure()
    ax = plt.gca()
    for key in evaluation_dict:
        parts = key.split("_")
        label = f"$M_{{Z'}} = {parts[1]},  r_{{inv}} = 0.{parts[2].split('p')[1]}$"
        rinv_value = f"0.{parts[2].split('p')[1]}"
        m_z = parts[1]
        score_sig = []
        score_bkg = []
        weights_sig = []
        weights_bkg = []
        for label, score,weights in zip(evaluation_dict[key]["target_labels"], evaluation_dict[key]["scores"], evaluation_dict[key]["weights"]):
            if label == 1:
                score_sig.append(score)
                weights_sig.append(weights)
            elif (qcd == True) & (label == 0):
                score_bkg.append(score)
                weights_bkg.append(weights)
        if qcd == True:
            plt.hist(score_bkg, bins = 50 , color=colors["qcd"], linewidth = 0.9, linestyle='-', histtype = 'step', label="Bkg")
        plt.hist(score_sig, bins = 50 , color=colors[m_z], linewidth = 0.9, linestyle=line_styles[rinv_value], histtype = 'step', label=label)
        qcd = False

    # Create dummy plots for separate legends
    # Dummy plots for MZ values
    legend_handles_mz = [plt.Line2D([0], [1], color=color, linewidth = 0.9,linestyle='-', label=f"$M_{{Z'}} = {m_z}$") for m_z, color in colors.items()]
    # Dummy plots for rinv values
    legend_handles_rinv = [plt.Line2D([0], [1], color='black',linewidth = 0.9, linestyle=linestyle, label=f"$r_{{inv}} = {rinv}$") for rinv, linestyle in line_styles.items()]

    additional_handle_qcd = plt.Line2D([0], [1], color='dimgrey', linewidth=0.9, linestyle='-', label='Bkg')

    # Combine all legend handles
    legend_handles_rinv = legend_handles_rinv + [additional_handle_qcd]

    # Plot the scores in a histogram
    plt.xlabel("Score")
    plt.ylabel("Number of Graphs")

    add_legends(ax,[legend_handles_mz,legend_handles_rinv])
    plt.grid()

    output_path = output_directory(name)
    outout_plot_name = name + suffix +"_score.png"
    plt.savefig(os.path.join(output_path,outout_plot_name), dpi=400)
    plt.close()


    #Again but normalized!
    qcd = True
    plt.figure()
    ax = plt.gca()
    for key in evaluation_dict:
        parts = key.split("_")
        label = f"$M_{{Z'}} = {parts[1]},  r_{{inv}} = 0.{parts[2].split('p')[1]}$"
        rinv_value = f"0.{parts[2].split('p')[1]}"
        m_z = parts[1]
        score_sig = []
        score_bkg = []
        weights_sig = []
        weights_bkg = []
        for label, score, weights in zip(evaluation_dict[key]["target_labels"], evaluation_dict[key]["scores"], evaluation_dict[key]["weights"]):
            if label == 1:
                score_sig.append(score)
                weights_sig.append(weights)
            elif (qcd == True) & (label == 0):
                score_bkg.append(score)
                weights_bkg.append(weights)
        if qcd == True:
            plt.hist(score_bkg,  bins = 50 , density=True, color=colors["qcd"], linewidth = 0.9, linestyle='-', histtype = 'step', label="Bkg")
        plt.hist(score_sig, bins = 50 ,density=True, color=colors[m_z], linewidth = 0.9, linestyle=line_styles[rinv_value], histtype = 'step', label=label)
        qcd = False

    # Create dummy plots for separate legends
    # Dummy plots for MZ values
    legend_handles_mz = [plt.Line2D([0], [1], color=color, linewidth = 0.9,linestyle='-', label=f"$M_{{Z'}} = {m_z}$") for m_z, color in colors.items()]
    # Dummy plots for rinv values
    legend_handles_rinv = [plt.Line2D([0], [1], color='black',linewidth = 0.9, linestyle=linestyle, label=f"$r_{{inv}} = {rinv}$") for rinv, linestyle in line_styles.items()]

    additional_handle_qcd = plt.Line2D([0], [1], color='dimgrey', linewidth=0.9, linestyle='-', label='Bkg')

    # Combine all legend handles
    legend_handles_rinv = legend_handles_rinv + [additional_handle_qcd]

    # Plot the scores in a histogram
    plt.xlabel("Score")
    plt.ylabel("Number of Graphs")

    # Create the first legend for MZ values            canvas = ROOT.TCanvas("canvas", "canvas", 1200, 1200)

    add_legends(ax,[legend_handles_mz,legend_handles_rinv])

    plt.grid()
    plt.yscale('log')

    output_path = output_directory(name)
    outout_plot_name = name + suffix +"_score_normalized.png"
    plt.savefig(os.path.join(output_path,outout_plot_name), dpi=400)
    plt.close()

def plot_dnn(evaluation_dict, name, suffix = "_eval"):
    #Plot all signals vs Bkg
    bkg = True
    score_sig = []
    score_bkg = []
    weights_sig = []
    weights_bkg = []
    for key in evaluation_dict:

        for label, score, weights in zip(evaluation_dict[key]["target_labels"], evaluation_dict[key]["scores"], evaluation_dict[key]["weights"]):
            if label == 1:
                score_sig.append(score)
                weights_sig.append(weights)
            elif (bkg == True) & (label == 0):
                score_bkg.append(score)
                weights_bkg.append(weights)
        
        if bkg == True:
            plt.hist(score_bkg,bins = 50 , color='#457B9D', density=True,  linewidth = 0.9, linestyle='-', histtype = 'step', label="all Bkg")
        bkg = False

    plt.hist(score_sig,bins = 50 , color='firebrick', density=True, linewidth = 0.9, linestyle='-', histtype = 'step', label="all Signals")
    
    plt.legend()

    # Plot the scores in a histogram
    plt.xlabel("GNN Score")
    plt.ylabel("Number of Graphs")


    plt.grid()
    plt.yscale('log')

    output_path = output_directory(name)
    outout_plot_name = name + suffix +"_DNN_Score.png"
    plt.savefig(os.path.join(output_path,outout_plot_name), dpi=400)
    plt.close()

def plot_dnn_2(evaluation_dict, name, suffix = "_eval"):
    #Plot all signals vs Bkg ( TTjet and QCD singlely)

    qcd = True
    ttjet = True
    ew = True
    score_sig = []
    score_qcd = []
    score_ttjet = []
    score_ew = []
    weights_sig = []
    weights_qcd = []
    weights_ttjet = []
    weights_ew = []
    for key in evaluation_dict:

        for label, score, dataset_label, weights in zip(evaluation_dict[key]["target_labels"], evaluation_dict[key]["scores"],evaluation_dict[key]["dataset_labels"], evaluation_dict[key]["weights"]):
            if label == 1:
                score_sig.append(score)
                weights_sig.append(weights)
            elif (qcd == True) & ("QCD" in dataset_label):
                score_qcd.append(score)
                weights_qcd.append(weights)
            elif (ttjet == True) & ("TTJet" in dataset_label):
                score_ttjet.append(score)
                weights_ttjet.append(weights)
            elif (ew == True) & (("WJets" in dataset_label) or ("JetsToNuNu" in dataset_label)):
                score_ew.append(score)
                weights_ew.append(score)
        
        if qcd == True:
            plt.hist(score_qcd,bins = 50 , color='#457B9D', density=True,  linewidth = 0.9, linestyle='-', histtype = 'step', label="QCD")
        qcd = False

        if ttjet == True:
            plt.hist(score_ttjet,bins = 50 , color='#A8DADC', density=True,  linewidth = 0.9, linestyle='-', histtype = 'step', label="TTJet")
        ttjet = False

        if ew == True:
            plt.hist(score_ew,bins = 50 , color='mediumaquamarine', density=True,  linewidth = 0.9, linestyle='-', histtype = 'step', label="EW")
        ew = False

    plt.hist(score_sig,bins = 50 , color='firebrick', density=True, linewidth = 0.9, linestyle='-', histtype = 'step', label="all Signals")
    plt.legend()

    # Plot the scores in a histogram
    plt.xlabel("GNN Score")
    plt.ylabel("Number of Graphs")


    plt.grid()
    plt.yscale('log')

    output_path = output_directory(name)
    outout_plot_name = name + suffix +"_DNN_Score_2.png"
    plt.savefig(os.path.join(output_path,outout_plot_name), dpi=400)
    plt.close()


def add_legends(ax, legends, loc='upper right', spacing=0.26):
    """
    Add multiple legends to a Matplotlib axes.

    Parameters:
        ax: The axes to which legends are added.
        legends: A list of legend handles, each a list of Line2D objects to be included in a legend.
        loc: Location of the legends (default 'upper right').
        spacing: Horizontal space between legends (default 0.1).
    """
    num_legends = len(legends)
    base_anchor = (0.995, 0.995)

    
    for i, handles in enumerate(legends):
        # Adjust the x-coordinate of the bbox_to_anchor for each legend
        x_offset = base_anchor[0] - i * spacing
        legend = ax.legend(handles=handles, loc=loc, bbox_to_anchor=(x_offset, base_anchor[1]))
        ax.add_artist(legend)
    
def plot_sic(roc, name, suffix = "_eval"):
    """Plot SIC for all Mediator masses and rinv in one plot

    Args:
        roc(dict): a dict created in roc_auc() -> roc[key = different signal hypotheses] = {"fpr": fpr_,"tpr": tpr_,"threshold": thres_}
        name(str): Name of the training for saving the png
       

    Returns:
    """

    line_styles = {
        '0.3': '-',
        '0.5': '--',
        '0.7': ':'
    }

    colors = {
        '700': 'rebeccapurple',        
        '800': 'plum',     
        '900': 'lightcoral',    
        '1000': 'firebrick',        
        '1100': 'gold',      
        '1200': 'olive',     
        '1300': 'yellowgreen',     
        '1400': '#A8DADC',      # powder blue
        '1500': '#457B9D',    # Steel blue
        '3000': '#1D3557',    # Dark blue
    }

    
    plt.figure()
    ax = plt.gca()

    # Iterate over each key in roc dictionary and plot the curves
    for key in roc:
        parts = key.split("_")
        label = f"$M_{{Z'}} = {parts[1]},  r_{{inv}} = 0.{parts[2].split('p')[1]}$"
        rinv_value = f"0.{parts[2].split('p')[1]}"
        m_z = parts[1]

        tpr = roc[key]["tpr"]
        sigma = roc[key]["sic"]


        plt.plot(tpr, sigma, color=colors[m_z], linewidth = 0.9, linestyle=line_styles[rinv_value], label=label)

    # Create dummy plots for separate legends
    # Dummy plots for MZ values
    legend_handles_mz = [plt.Line2D([0], [1], color=color, linewidth = 0.9,linestyle='-', label=f"$M_{{Z'}} = {m_z}$") for m_z, color in colors.items()]
    # Dummy plots for rinv values
    legend_handles_rinv = [plt.Line2D([0], [1], color='black',linewidth = 0.9, linestyle=linestyle, label=f"$r_{{inv}} = {rinv}$") for rinv, linestyle in line_styles.items()]

    # Plot the actual ROC curves
    plt.ylabel("SIC")
    plt.xlabel("True Positive Rate")
    # plt.ylim(0,7)
    plt.axhline(y=1, color='gray', linestyle='--')
    #plt.yscale('log')

    add_legends(ax,[legend_handles_mz,legend_handles_rinv])
    plt.grid()

    output_path = output_directory(name)
    outout_plot_name = name +suffix + "_SIC.png"
    plt.savefig(os.path.join(output_path,outout_plot_name), dpi=400)
    plt.close()
    print(f"{outout_plot_name} has been saved.")

    
def plot_sic_small(roc, name, suffix = "_eval"):
    """Plot SIC for a small choise of Mediator masses and rinv in one plot

    Args:
        roc(dict): a dict created in roc_auc() -> roc[key = different signal hypotheses] = {"fpr": fpr_,"tpr": tpr_,"threshold": thres_}
        name(str): Name of the training for saving the png
       

    Returns:
    """

    line_styles = {
        '0.3': '-',
        '0.5': '--',
        '0.7': ':'
    }

    colors = {
        '700': 'plum',        # Plum
        '900': '#E63946',     # Firebrick
        '1100': 'gold',    
        '1300': '#457B9D',    # Steel blue
        '1500': '#1D3557',    # Dark blue
    }
    
    # List of numbers to check for in the key
    numbers_to_check = ["800", "1000", "1200", "1400", "3000"]
    
    plt.figure()
    ax = plt.gca()

    # Iterate over each key in roc dictionary and plot the curves
    for key in roc:
        if any(num in key for num in numbers_to_check):
            continue
        parts = key.split("_")
        label = f"$M_{{Z'}} = {parts[1]},  r_{{inv}} = 0.{parts[2].split('p')[1]}$"
        rinv_value = f"0.{parts[2].split('p')[1]}"
        m_z = parts[1]

        tpr = roc[key]["tpr"]
        sigma = roc[key]["sic"]


        plt.plot(tpr, sigma, color=colors[m_z], linewidth = 0.9, linestyle=line_styles[rinv_value], label=label)

    # Create dummy plots for separate legends
    # Dummy plots for MZ values
    legend_handles_mz = [plt.Line2D([0], [1], color=color, linewidth = 0.9,linestyle='-', label=f"$M_{{Z'}} = {m_z}$") for m_z, color in colors.items()]
    # Dummy plots for rinv values
    legend_handles_rinv = [plt.Line2D([0], [1], color='black',linewidth = 0.9, linestyle=linestyle, label=f"$r_{{inv}} = {rinv}$") for rinv, linestyle in line_styles.items()]

    # Plot the actual ROC curves
    plt.ylabel("SIC")
    plt.xlabel("True Positive Rate")
    # plt.ylim(0,7)
    plt.axhline(y=1, color='gray', linestyle='--')
    #plt.yscale('log')

    plt.grid()

    add_legends(ax,[legend_handles_mz,legend_handles_rinv])

    output_path = output_directory(name)
    outout_plot_name = name +suffix + "_SIC_small.png"
    plt.savefig(os.path.join(output_path,outout_plot_name), dpi=400)
    plt.close()
    print(f"{outout_plot_name} has been saved.")