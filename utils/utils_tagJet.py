
import numpy as np
import awkward as ak
import pickle
import os


def get_threshold_from_sic(name, file_name='_eval_ROC_AUC.pkl'):
    """Get the threshold to cut from the maximal sic value
    Args:
        name(str): Name of the Network which we take
        file_name(str): Which sic we want to use: either from _eval_ROC_AUC.pkl ( jet level) or Eventlevel_ROC.pkl( event level)
 
    Returns:   
        threshold (float): threshold where to place the cut
        threshold (Dict[float]): threshold for max SIC for each signal hypothesis
        sic (Dict[float]): SIC for this threshold
        tpr (Dict[float]): tpr for this threshold
        fpr (Dict[float]): fpr for this threshold
    """
    directory = '/t3home/rhampp/SVJ_GNN/output'

    file_to_open = os.path.join(directory, name, file_name)
    with open(file_to_open, 'rb') as file:
        evaluation_dict = pickle.load(file) 
        print(f"Opened file: {file_to_open}")

    wp = {}
    tpr = {}
    sic = {}


    for key in evaluation_dict:
        tpr[key] = evaluation_dict[key]["tpr"]
        fpr = evaluation_dict[key]["fpr"]
        threshold = evaluation_dict[key]["threshold"]

        sic[key] = evaluation_dict[key]["sic"]

        max_sic = max(sic[key])
        max_index = np.argmax(sic[key])
        wp[key] = threshold[max_index]

    wp_av = sum(wp.values()) / len(wp)
    
    print("Maximal average SIC for a cut at:", wp_av)

    return wp_av, wp, max_sic, tpr, fpr


def roc_event_level(GNN_score_sig, GNN_score_bkg):
    """Get roc on event level: TPR: probablilty of two SVJ in signal, FPR probability of two SVJ in background
    Args:
        GNN_score_sig(List[score1,score2]): Score of the first two jets of the signal
        GNN_score_bkg(List[score1,score2]): Score of the first two jets of the background
 
    Returns:   
        tpr (array[float]): True positiv rate
        fpr (array[float]): False positiv rate
        threshold (array[float]): threshold where to place the cut
    """
    # Convert input lists to Awkward Arrays
    GNN_score_sig = ak.Array(GNN_score_sig)
    GNN_score_bkg = ak.Array(GNN_score_bkg)
    
    # Precompute the lengths
    len_sig = len(GNN_score_sig)
    len_bkg = len(GNN_score_bkg)

    # Define thresholds
    thresholds = np.arange(0, 1, 0.01)

    # Initialize lists for TPR, FPR, and thresholds
    tpr = []
    fpr = []

    # Iterate over each threshold
    for thres in thresholds:
        # Compute number of signals and backgrounds above threshold
        n_sig = np.sum((ak.all(GNN_score_sig > thres, axis=1)) & (ak.num(GNN_score_sig, axis=1) == 2))
        n_bkg = np.sum((ak.all(GNN_score_bkg > thres, axis=1)) & (ak.num(GNN_score_bkg, axis=1) == 2))

        # Calculate TPR and FPR
        tpr.append(n_sig / len_sig)
        fpr.append(n_bkg / len_bkg)

    return np.array(tpr), np.array(fpr), thresholds











    















    
