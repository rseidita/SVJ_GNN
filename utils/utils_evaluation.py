from sklearn.metrics import roc_curve, roc_auc_score
import numpy as np
from data_loading.graphDataLoader import GraphDataLoader
from TrainingandValidation import output_directory
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import re
import pickle
import os
from itertools import chain

from TrainingandValidation import output_directory

def Evaluate_model(base_directory_QCD, base_directory_SVJ, model_type, eval_type, batch_size, normalizer, model, device,window_to_load= [0.7,1], jet_idx=None):
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
        window_to_load: for the evaluation: standard = [0.7,1]
        jet_idx(int): Which jet to look at (0,1,2), default = None -> look at all graphs in the window


    Returns:   evaluation_dict[SVJ_***_*p*] = {
                'weights': np.concatenate(weights),
                'target_labels': np.concatenate(target_labels),
                'scores': np.concatenate(test_scores)}
        
    """


 
    datasets = [
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

            [base_directory_SVJ + "SVJ_mMed-1500GeV_mDark-20GeV_rinv-0.3_alpha-peak", f"SVJ_1500_0p3_{model_type}{eval_type}"],
            [base_directory_SVJ + "SVJ_mMed-1500GeV_mDark-20GeV_rinv-0.5_alpha-peak", f"SVJ_1500_0p5_{model_type}{eval_type}"],
            [base_directory_SVJ + "SVJ_mMed-1500GeV_mDark-20GeV_rinv-0.7_alpha-peak", f"SVJ_1500_0p7_{model_type}{eval_type}"],

            [base_directory_SVJ + "SVJ_mMed-3000GeV_mDark-20GeV_rinv-0.3_alpha-peak", f"SVJ_3000_0p3_{model_type}{eval_type}"],
            [base_directory_SVJ + "SVJ_mMed-3000GeV_mDark-20GeV_rinv-0.5_alpha-peak", f"SVJ_3000_0p5_{model_type}{eval_type}"],
            [base_directory_SVJ + "SVJ_mMed-3000GeV_mDark-20GeV_rinv-0.7_alpha-peak", f"SVJ_3000_0p7_{model_type}{eval_type}"],
        ]

    evaluation_dict = {}
    dataloader_bkg = GraphDataLoader(
            [[base_directory_QCD , f"QCD_{model_type}{eval_type}"]],
            window_to_load=[window_to_load for _ in range(1)],
            batch_size=batch_size,
            normalizer=normalizer,
            jet_idx=jet_idx,
        )

    for dataset in datasets:
        dataloader_sig = GraphDataLoader(
            [dataset],
            window_to_load=[window_to_load for _ in range(1)],
            batch_size=batch_size,
            normalizer=normalizer,
            jet_idx=jet_idx,
        )


        weights = []
        target_labels = []
        test_scores = []
        
        model.eval()


        for graphs in chain(dataloader_bkg,dataloader_sig):
            weights.append(np.array(graphs.w))
            graphs = graphs.to(device)
            target = graphs.y
            out = model(graphs)
            score = out[:,0].cpu()
            target_labels.append(np.array(target.cpu()))
            test_scores.append(score.detach().numpy())

        evaluation_dict[dataset[1]] = {
            'weights': np.concatenate(weights),
            'target_labels': np.concatenate(target_labels),
            'scores': np.concatenate(test_scores)
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

        if np.all(target == 0) or np.all(target == 1):
            continue

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
            "sic": sic_
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

    Returns:  auc(dict) with the form: {(Mz, rinv): 0.6106412792103517, ...
        
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


def plot_roc(roc, name, suffix = "_eval"):

    # Define line styles for each value of rinv
    line_styles = {
        '0.3': '-',
        '0.5': '--',
        '0.7': ':'
    }

    colors = {
        '700': 'plum',        # Plum
        '800': '#E63946',     # Firebrick
        '900': '#F4A261',     # Tangerine yellow
        '1000': '#A8DADC',    # Powder blue
        '1500': '#457B9D',    # Steel blue
        '3000': '#1D3557',    # Dark blue
    }

 
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
    first_legend = plt.legend(handles=legend_handles_mz, loc=[0.735,0.02])
    ax = plt.gca().add_artist(first_legend)

    plt.legend(handles=legend_handles_rinv, loc=[0.5,0.02])
    plt.grid()

    output_path = output_directory(name)
    outout_plot_name = name + suffix + "_roc.png"
    print(f"{outout_plot_name} has been saved.")
    plt.savefig(os.path.join(output_path,outout_plot_name), dpi=800)
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
    plt.figure(figsize=(10, 10))
    im  = plt.imshow(gridxy, cmap=new_cmap, interpolation='nearest')
    # Show value auc in each bin
    for idx_x, x in enumerate(x_list):
        for idx_y, y in enumerate(y_list):
            plt.text(idx_x, idx_y, "{:.3f}".format(gridxy[idx_y][idx_x]), va='center', ha='center', color='white', fontsize=22)
    
    if (variable_name == "NormSignalCont"):
        im.set_clim(0.001, 0.4)
    if (variable_name == "AUC"):
        im.set_clim(0.5, 1.0)
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label(variable_name)
    
    plt.xticks(np.arange(len(x_list)), x_list)
    plt.yticks(np.arange(len(y_list)), y_list)
    plt.ylabel("$r_{inv}$")
    plt.xlabel("$M_{Z'}$")

    #hep.cms.label('Preliminary', data=False)
    #hep.cms.label(loc=0)

    output_path = output_directory(name)
    outout_plot_name = name + suffix + "_auc.png"
    print(f"{outout_plot_name} has been saved.")
    plt.savefig(os.path.join(output_path,outout_plot_name),  dpi=800)
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

    colors = {
        '700': 'plum',        # Plum
        '800': '#E63946',     # Firebrick
        '900': '#F4A261',     # Tangerine yellow
        '1000': '#A8DADC',    # Powder blue
        '1500': '#457B9D',    # Steel blue
        '3000': '#1D3557',    # Dark blue
        'qcd': 'dimgrey',
    }
 
    # Iterate over each key in roc dictionary and add the scores
    qcd = True
    for key in evaluation_dict:
        parts = key.split("_")
        label = f"$M_{{Z'}} = {parts[1]},  r_{{inv}} = 0.{parts[2].split('p')[1]}$"
        rinv_value = f"0.{parts[2].split('p')[1]}"
        m_z = parts[1]
        score_sig = []
        score_bkg = []
        for label, score in zip(evaluation_dict[key]["target_labels"], evaluation_dict[key]["scores"]):
            if label == 1:
                score_sig.append(score)
            elif (qcd == True) & (label == 0):
                score_bkg.append(score)
        if qcd == True:
            plt.hist(score_bkg,bins = 50 , color=colors["qcd"], linewidth = 0.9, linestyle='-', histtype = 'step', label="QCD")
        plt.hist(score_sig,bins = 50 , color=colors[m_z], linewidth = 0.9, linestyle=line_styles[rinv_value], histtype = 'step', label=label)
        qcd = False

    # Create dummy plots for separate legends
    # Dummy plots for MZ values
    legend_handles_mz = [plt.Line2D([0], [1], color=color, linewidth = 0.9,linestyle='-', label=f"$M_{{Z'}} = {m_z}$") for m_z, color in colors.items()]
    # Dummy plots for rinv values
    legend_handles_rinv = [plt.Line2D([0], [1], color='black',linewidth = 0.9, linestyle=linestyle, label=f"$r_{{inv}} = {rinv}$") for rinv, linestyle in line_styles.items()]

    additional_handle_qcd = plt.Line2D([0], [1], color='dimgrey', linewidth=0.9, linestyle='-', label='QCD')

    # Combine all legend handles
    legend_handles_rinv = legend_handles_rinv + [additional_handle_qcd]

    # Plot the scores in a histogram
    plt.xlabel("Score")
    plt.ylabel("Number of Graphs")

    # Create the first legend for MZ values
    first_legend = plt.legend(handles=legend_handles_mz, loc=[0.735,0.59])
    ax = plt.gca().add_artist(first_legend)

    plt.legend(handles=legend_handles_rinv, loc=[0.5,0.76])
    plt.grid()

    output_path = output_directory(name)
    outout_plot_name = name + suffix +"_score.png"
    plt.savefig(os.path.join(output_path,outout_plot_name), dpi=800)
    plt.close()

    #Again but normalized!
    qcd = True
    for key in evaluation_dict:
        parts = key.split("_")
        label = f"$M_{{Z'}} = {parts[1]},  r_{{inv}} = 0.{parts[2].split('p')[1]}$"
        rinv_value = f"0.{parts[2].split('p')[1]}"
        m_z = parts[1]
        score_sig = []
        score_bkg = []
        for label, score in zip(evaluation_dict[key]["target_labels"], evaluation_dict[key]["scores"]):
            if label == 1:
                score_sig.append(score)
            elif (qcd == True) & (label == 0):
                score_bkg.append(score)
        if qcd == True:
            plt.hist(score_bkg,bins = 50 , color=colors["qcd"],density=True,  linewidth = 0.9, linestyle='-', histtype = 'step', label="QCD")
        plt.hist(score_sig,bins = 50 , color=colors[m_z], density=True, linewidth = 0.9, linestyle=line_styles[rinv_value], histtype = 'step', label=label)
        qcd = False

    # Create dummy plots for separate legends
    # Dummy plots for MZ values
    legend_handles_mz = [plt.Line2D([0], [1], color=color, linewidth = 0.9,linestyle='-', label=f"$M_{{Z'}} = {m_z}$") for m_z, color in colors.items()]
    # Dummy plots for rinv values
    legend_handles_rinv = [plt.Line2D([0], [1], color='black',linewidth = 0.9, linestyle=linestyle, label=f"$r_{{inv}} = {rinv}$") for rinv, linestyle in line_styles.items()]

    additional_handle_qcd = plt.Line2D([0], [1], color='dimgrey', linewidth=0.9, linestyle='-', label='QCD')

    # Combine all legend handles
    legend_handles_rinv = legend_handles_rinv + [additional_handle_qcd]

    # Plot the scores in a histogram
    plt.xlabel("Score")
    plt.ylabel("Number of Graphs")

    # Create the first legend for MZ values            canvas = ROOT.TCanvas("canvas", "canvas", 1200, 1200)

    first_legend = plt.legend(handles=legend_handles_mz, loc=[0.735,0.59])
    ax = plt.gca().add_artist(first_legend)

    plt.legend(handles=legend_handles_rinv, loc=[0.5,0.76])
    plt.grid()

    output_path = output_directory(name)
    outout_plot_name = name + suffix +"_score_normalized.png"
    plt.savefig(os.path.join(output_path,outout_plot_name), dpi=800)
    plt.close()



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
        '700': 'plum',        # Plum
        '800': '#E63946',     # Firebrick
        '900': '#F4A261',     # Tangerine yellow
        '1000': '#A8DADC',    # Powder blue
        '1500': '#457B9D',    # Steel blue
        '3000': '#1D3557',    # Dark blue
    }
 
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
    plt.ylim(0,30)
    plt.axhline(y=1, color='gray', linestyle='--')
    #plt.yscale('log')

    # Create the first legend for MZ values
    first_legend = plt.legend(handles=legend_handles_mz, loc=[0.735,0.62])
    ax = plt.gca().add_artist(first_legend)

    plt.legend(handles=legend_handles_rinv, loc=[0.5,0.79])
    plt.grid()

    output_path = output_directory(name)
    outout_plot_name = name +suffix + "_SIC.png"
    plt.savefig(os.path.join(output_path,outout_plot_name), dpi=800)
    plt.close()
    print(f"{outout_plot_name} has been saved.")

    

    
    
    

