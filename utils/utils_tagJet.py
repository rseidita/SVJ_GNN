
import numpy as np
import numpy as np
import pickle
import os


def get_threshold_from_sic(name):
    """Get the threshold to cut from the maximal sic value
    Args:
        name(str): Name of the Network which we take
 
    Returns:   
        threshold (float): threshold where to place the cut
        threshold (Dict[float]): threshold for max SIC for each signal hypothesis
        sic (Dict[float]): SIC for this threshold
        tpr (Dict[float]): tpr for this threshold
        fpr (Dict[float]): fpr for this threshold
    """
    directory = '/t3home/rhampp/SVJ_GNN/output'
    file_name = '_eval_ROC_AUC.pkl'

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













    















    
