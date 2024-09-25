
import numpy as np
import awkward as ak
import pickle
import os


def get_threshold_from_sic(name, file_name='_eval_ROC_AUC.pkl', max_confined = False):
    """Get the threshold to cut from the maximal sic value
    Args:
        name(str): Name of the Network which we take
        file_name(str): Which sic we want to use: either from _eval_ROC_AUC.pkl ( jet level) or Eventlevel_ROC.pkl( event level)
        max_confined(bool): Confine the search of the maximum SIC to a certain region (tpr within [0.2,1])
    Returns:   
        threshold (float): threshold where to place the cut
        threshold (Dict[float]): threshold for max SIC for each signal hypothesis
        sic (Dict[float]): SIC for this threshold
        tpr_max (Dict[float]): tpr for this threshold
        fpr_max (Dict[float]): fpr for this threshold
        tpr_tot (float): tpr for all signals toghether for this threshold
        fpr_av (float): fpr average for all signals for this threshold

    """
    directory = '/t3home/rhampp/SVJ_GNN/output'

    file_to_open = os.path.join(directory, name, file_name)
    with open(file_to_open, 'rb') as file:
        evaluation_dict = pickle.load(file) 
        print(f"Opened file: {file_to_open}")

    wp = {}
    tpr = {}
    fpr = {}
    sic = {}
    max_sic = {}
    tpr_max = {}
    fpr_max = {}
    threshold = {}


    # Iterate over each key in evaluation_dict
    for key in evaluation_dict:
        # Extract TPR, FPR, threshold, and SIC values
        tpr[key] = evaluation_dict[key]["tpr"]
        fpr[key] = evaluation_dict[key]["fpr"]
        threshold[key] = evaluation_dict[key]["threshold"]
        sic[key] = evaluation_dict[key]["sic"]
       
        # Check if filtered_sic is not empty
        if max_confined:
            # Define the range for TPR
            tpr_min = 0.1
            tpr_max_range = 1

            # Create a mask for TPR values within the specified range
            tpr_mask = (tpr[key] >= tpr_min) & (tpr[key] <= tpr_max_range)
            filtered_sic = sic[key][tpr_mask]

            # Calculate the maximum SIC and its corresponding index within the filtered range
            max_sic[key] = max(filtered_sic)
            max_index = np.argmax(filtered_sic)
            
            # Find the corresponding threshold, TPR, and FPR for the max SIC
            filtered_threshold = threshold[key][tpr_mask]
            filtered_tpr = tpr[key][tpr_mask]
            filtered_fpr = fpr[key][tpr_mask]

            wp[key] = filtered_threshold[max_index]
            tpr_max[key] = filtered_tpr[max_index]
            fpr_max[key] = filtered_fpr[max_index]

        else:
            max_sic[key] = max(sic[key])
            max_index = np.argmax(sic[key])
            wp[key] = threshold[key][max_index]
            tpr_max[key] = tpr[key][max_index] 
            fpr_max[key] = fpr[key][max_index] 

    

    wp_av = sum(wp.values()) / len(wp)
    
    print("Maximal average SIC for a cut at:", wp_av)
    print("tpr = ", tpr_max)
    print("fpr = ", fpr_max)
    print("SIC = ", max_sic)

    tpr_max, fpr_max = get_tpr_for_threshold(wp_av, evaluation_dict)

    tpr_tot, fpr_av = get_tpr_for_all_signals(wp_av, evaluation_dict)

    return wp_av, wp, max_sic, tpr_max, fpr_max, tpr_tot, fpr_av

def find_closest(arr, target):

    # Initialize variables to store the closest value and its index
    closest_value = None
    closest_index = None
    min_diff = float('inf')
    
    # Iterate over the array
    for index, value in enumerate(arr):
        # Calculate the difference between the current value and the target
        diff = abs(value - target)
        
        # If this difference is smaller than the current smallest difference
        if diff < min_diff:
            min_diff = diff
            closest_value = value
            closest_index = index
    
    return closest_value, closest_index

def get_tpr_for_threshold(wp, evaluation_dict):
    """Get the tpr and fpr for a certain threshold
    Args:
        wp (float): Threshold
        evaluation_dict(dict): dict with tpr and fpr for different signal hypothesis
 
    Returns:   
        tpr (Dict[float]): tpr for this threshold
        fpr (Dict[float]): fpr for this threshold
    """


    tpr = {}
    fpr = {}


    for key in evaluation_dict:
        threshold, closest_index = find_closest(evaluation_dict[key]["threshold"], wp)
        tpr[key] = evaluation_dict[key]["tpr"][closest_index]
        fpr[key] = evaluation_dict[key]["fpr"][closest_index]

    return tpr, fpr
    

def get_tpr_for_all_signals(wp, evaluation_dict):
    """Get the tpr for a certain threshold for all signals toghether and fpr average
    Args:
        wp (float): Threshold
        evaluation_dict(dict): dict with tpr and fpr for different signal hypothesis
 
    Returns:   
        tpr (Dict[float]): tpr for this threshold
        fpr (Dict[float]): fpr for this threshold
    """

    tpr = {}
    fpr = {}
    # n_tot = [1140,907,473,9346,5952,2188,11744,7751,2921,13941,9577,3676,15584,11092,4327,3321,2038,792,21300,18048,7068,3210,1944,824,6253,3806,1462,10677,10410,6047]
    n_tot = [1049,838,433,2617,1596,684,5149,3124,1217,2463,1512,592,9346,5952,2188,11744,7751,2921,13941,9577,3676,15584,11092,4327,19555,16661,6475,5271,5134,2984]

    for key in evaluation_dict:
        threshold, closest_index = find_closest(evaluation_dict[key]["threshold"], wp)
        tpr[key] = evaluation_dict[key]["tpr"][closest_index]
        fpr[key] = evaluation_dict[key]["fpr"][closest_index]
    print(tpr)
    print(fpr)
    tpr_tot = sum(tpr[key]* n  for key, n in zip(tpr, n_tot)) / sum(n_tot) 
    fpr_tot = sum(fpr[key] for key in fpr)/len(fpr)

    return tpr_tot, fpr_tot

    


def roc_event_level(GNN_score_sig, GNN_score_bkg, weights_sig, weights_bkg):
    """Get roc on event level: TPR: probablilty of two SVJ in signal, FPR probability of two SVJ in background
    Args:
        GNN_score_sig(List[score1,score2]): Score of the first two jets of the signal
        GNN_score_bkg(List[score1,score2]): Score of the first two jets of the background
        weights_sig (List[float]): Weights for the signal events
        weights_bkg (List[float]): Weights for the background events
 
    Returns:   
        tpr (array[float]): True positiv rate
        fpr (array[float]): False positiv rate
        threshold (array[float]): threshold where to place the cut
    """
    # Convert input lists to Awkward Arrays
    GNN_score_sig = ak.Array(GNN_score_sig)
    GNN_score_bkg = ak.Array(GNN_score_bkg)
    weights_sig = ak.Array(weights_sig)
    weights_bkg = ak.Array(weights_bkg)

    # Define thresholds
    thresholds = np.arange(0, 1, 0.01)

    # Initialize lists for TPR, FPR, and thresholds
    tpr = []
    fpr = []

    # Precompute the total weights
    total_weight_sig = np.sum(weights_sig)
    total_weight_bkg = np.sum(weights_bkg)

    # Iterate over each threshold
    for thres in thresholds:
        # Compute weighted number of signals and backgrounds above threshold
        n_sig = np.sum(weights_sig * ((ak.all(GNN_score_sig > thres, axis=1)) & (ak.num(GNN_score_sig, axis=1) == 2)))
        n_bkg = np.sum(weights_bkg * ((ak.all(GNN_score_bkg > thres, axis=1)) & (ak.num(GNN_score_bkg, axis=1) == 2)))

        # Calculate TPR and FPR
        tpr.append(n_sig / total_weight_sig)
        fpr.append(n_bkg / total_weight_bkg)

    return np.array(tpr), np.array(fpr), thresholds



def filter_by_svjtagger(sample, weights):
    """
    Filter the dictionarys into three categories based on the SVJTagger values.
    
    Args:
    - dict (dict): Dictionary of arrays in the following form: Dict={key = rinv/HT: FatJet_SVJTagger, MT01...}
    
    Returns:
    - dict: Three filtered dictionaries.
    """
    sample_0SVJ = {}
    sample_1SVJ = {}
    sample_2SVJ = {}
    
    weights_0SVJ = {}
    weights_1SVJ = {}
    weights_2SVJ = {}


    for key in sample:
        ak_array = ak.Array(sample[key])
        ak_array_weights = ak.Array(weights[key])

        mask_0SVJ = ak.all(ak_array.SVJ[:,:2] == False, axis=1)
        mask_1SVJ = ak.sum(ak_array.SVJ[:,:2] == True, axis=1) == 1
        mask_2SVJ = ak.all(ak_array.SVJ[:,:2] == True, axis=1)

        sample_0SVJ[key] = ak_array[mask_0SVJ]
        sample_1SVJ[key] = ak_array[mask_1SVJ]
        sample_2SVJ[key] = ak_array[mask_2SVJ]

        weights_0SVJ[key] = ak_array_weights[mask_0SVJ]
        weights_1SVJ[key] = ak_array_weights[mask_1SVJ]
        weights_2SVJ[key] = ak_array_weights[mask_2SVJ]

 

    return sample_0SVJ, sample_1SVJ, sample_2SVJ, weights_0SVJ, weights_1SVJ, weights_2SVJ







    















    
