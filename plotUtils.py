import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import os
#import mplhep as hep


def plot_table(
        table_dict,
        variable_name,
        file_name,
        x_label,
        y_label,
    ):

        x_list = []
        y_list = []
        for key in table_dict.keys():
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
                if key in table_dict.keys():
                    value = table_dict[key]
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
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        #hep.cms.label('Preliminary', data=False)
        #hep.cms.label(loc=0)

        outout_plot_name = file_name + ".pdf"
        print(f"{outout_plot_name} has been saved.")
        plt.savefig(outout_plot_name)
        plt.close()


def plot_sic(roc, name, output_directory, suffix = "_eval"):
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
        '700': 'hotpink',        
        '800': 'mediumseagreen',     
        '900': 'gold',    
        '1000': 'darkorange',    
        '1100': 'darkgreen',
        '1200': 'rebeccapurple',
        '1300': 'deepskyblue',
        '1400': 'crimson',
        '1500': 'dodgerblue',    
        '3000': 'brown',    
    }
 
    # Iterate over each key in roc dictionary and plot the curves
    for key in roc:
        parts = key.split("_")
        label = f"$M_{{Z'}} = {parts[1]},  r_{{inv}} = 0.{parts[2].split('p')[1]}$"
        rinv_value = f"0.{parts[2].split('p')[1]}"
        m_z = parts[1]

        tpr = roc[key]["tpr"]
        fpr = roc[key]["fpr"]
        sigma = tpr/np.sqrt(fpr)

        plt.plot(tpr, sigma, color=colors[m_z], linewidth = 0.9, linestyle=line_styles[rinv_value], label=label)

    # Create dummy plots for separate legends
    # Dummy plots for MZ values
    legend_handles_mz = [plt.Line2D([0], [1], color=color, linewidth = 0.9,linestyle='-', label=f"$M_{{Z'}} = {m_z}$") for m_z, color in colors.items()]
    # Dummy plots for rinv values
    legend_handles_rinv = [plt.Line2D([0], [1], color='black',linewidth = 0.9, linestyle=linestyle, label=f"$r_{{inv}} = {rinv}$") for rinv, linestyle in line_styles.items()]

    # Plot the actual ROC curves
    plt.ylabel("SIC")
    plt.xlabel("True Positive Rate")
    # plt.ylim(0,30)
    plt.axhline(y=1, color='gray', linestyle='--')
    #plt.yscale('log')

    # Create the first legend for MZ values
    first_legend = plt.legend(handles=legend_handles_mz, loc=[0.735,0.62])
    ax = plt.gca().add_artist(first_legend)

    plt.legend(handles=legend_handles_rinv, loc=[0.5,0.79])
    plt.grid()

    output_path = output_directory + name
    outout_plot_name = name +suffix + "_SIC.png"
    plt.savefig(os.path.join(output_path,outout_plot_name), dpi=1200)
    plt.close()
    print(f"{outout_plot_name} has been saved.")


import ROOT
import uuid

def make_histograms(signals_dict, bkg_data, bkg_weights, signal_keys, nbins, file_name, suffix, hist_output_directory):
    
    unique_id = uuid.uuid4()
    name = f"c_{unique_id}"
    canvas = ROOT.TCanvas(name, "Canvas", 800, 600)

    sig0 = signals_dict[signal_keys[0]]
    sig1 = signals_dict[signal_keys[1]]
    sig2 = signals_dict[signal_keys[2]]
    sig3 = signals_dict[signal_keys[3]]
    sig4 = signals_dict[signal_keys[4]]

    all_data = [np.array(data) for data in signals_dict.values()] + [np.array(bkg_data)]

    xmin = min(arr.min() for arr in all_data)
    xmax = max(arr.max() for arr in all_data)

    hist0 = ROOT.TH1F(f"{signal_keys[0]}", f"{signal_keys[0]}", nbins, xmin, xmax)
    hist1 = ROOT.TH1F(f"{signal_keys[1]}", f"{signal_keys[1]}", nbins, xmin, xmax)
    hist2 = ROOT.TH1F(f"{signal_keys[2]}", f"{signal_keys[2]}", nbins, xmin, xmax)
    hist3 = ROOT.TH1F(f"{signal_keys[3]}", f"{signal_keys[3]}", nbins, xmin, xmax)
    hist4 = ROOT.TH1F(f"{signal_keys[4]}", f"{signal_keys[4]}", nbins, xmin, xmax)

    bkg_hist = ROOT.TH1F(f"bkg", f"bkg", nbins, xmin, xmax)

    bkg_hist.FillN(len(bkg_data), bkg_data, bkg_weights)

    hist0.FillN(len(sig0), sig0, np.ones_like(sig0))
    hist1.FillN(len(sig1), sig1, np.ones_like(sig1))
    hist2.FillN(len(sig2), sig2, np.ones_like(sig2))
    hist3.FillN(len(sig3), sig3, np.ones_like(sig3))
    hist4.FillN(len(sig4), sig4, np.ones_like(sig4))

    # Save histograms

    file_sig0 = ROOT.TFile(f"{hist_output_directory}/{signal_keys[0]}_{suffix}_hist.root", "RECREATE")
    hist0.Write()
    file_sig0.Close()

    file_sig1 = ROOT.TFile(f"{hist_output_directory}/{signal_keys[1]}_{suffix}_hist.root", "RECREATE")
    hist1.Write()
    file_sig1.Close()

    file_sig2 = ROOT.TFile(f"{hist_output_directory}/{signal_keys[2]}_{suffix}_hist.root", "RECREATE")
    hist2.Write()
    file_sig2.Close()

    file_sig3 = ROOT.TFile(f"{hist_output_directory}/{signal_keys[3]}_{suffix}_hist.root", "RECREATE")
    hist3.Write()
    file_sig3.Close()

    file_sig4 = ROOT.TFile(f"{hist_output_directory}/{signal_keys[4]}_{suffix}_hist.root", "RECREATE")
    hist4.Write()
    file_sig4.Close()

    file_bkg = ROOT.TFile(f"{hist_output_directory}/bkg_{suffix}_hist.root", "RECREATE")
    bkg_hist.Write()
    file_bkg.Close()

    canvas.Update()
    canvas.Close()


from sklearn.metrics import roc_auc_score, roc_curve


def calc_auc(signal_losses, bkg_losses):
    y_true = {}
    y_score = {}
    fpr = {}
    tpr = {}
    auc = {}


    for signal_name, losses in signal_losses.items():
        y_true[signal_name] = [0]*len(bkg_losses) + [1]*len(losses)
        y_score[signal_name] = bkg_losses + losses

        fpr[signal_name], tpr[signal_name], _ = roc_curve(y_true[signal_name], y_score[signal_name])
        auc[signal_name] = roc_auc_score(y_true[signal_name], y_score[signal_name])


    mean_auc = np.mean(list(auc.values()))

    return auc, mean_auc


def calc_roc_auc(signal_losses, bkg_losses):
    y_true = {}
    y_score = {}
    fpr = {}
    tpr = {}
    auc = {}
    thresholds = {}
    roc = {}

    for signal_name, losses in signal_losses.items():
        y_true[signal_name] = [0]*len(bkg_losses) + [1]*len(losses)
        y_score[signal_name] = bkg_losses + losses

        fpr[signal_name], tpr[signal_name], thresholds[signal_name] = roc_curve(y_true[signal_name], y_score[signal_name])
        auc[signal_name] = roc_auc_score(y_true[signal_name], y_score[signal_name])
        roc[signal_name] = {"fpr": fpr[signal_name],
                            "tpr": tpr[signal_name],
                            "threshold": thresholds[signal_name],}


    mean_auc = np.mean(list(auc.values()))

    return roc, auc

def plot_roc(signal_losses, bkg_losses, output_dir):

    line_styles = {
            '0.3': '-',
            '0.5': '--',
            '0.7': ':'
        }

    colors = {
        '700': 'hotpink',        
        '800': 'mediumseagreen',     
        '900': 'gold',    
        '1000': 'darkorange',    
        '1100': 'darkgreen',
        '1200': 'rebeccapurple',
        '1300': 'deepskyblue',
        '1400': 'crimson',
        '1500': 'dodgerblue',    
        '3000': 'brown',    
    }

    y_true = {}
    y_score = {}
    fpr = {}
    tpr = {}

    for signal_name, losses in signal_losses.items():
        y_true[signal_name] = [0]*len(bkg_losses) + [1]*len(losses)
        y_score[signal_name] = bkg_losses + losses

        fpr[signal_name], tpr[signal_name], _ = roc_curve(y_true[signal_name], y_score[signal_name])

    for signal_name in tpr.keys():
        
        parts = signal_name.split("_")
        label = f"$M_{{Z'}} = {parts[1]},  r_{{inv}} = 0.{parts[2].split('p')[1]}$"
        rinv_value = f"0.{parts[2].split('p')[1]}"
        m_z = parts[1]
        plt.plot(fpr[signal_name], tpr[signal_name], color=colors[m_z], linewidth = 0.9, linestyle=line_styles[rinv_value], label=label)

   
    # Create dummy plots for separate legends
    # Dummy plots for MZ values
    legend_handles_mz = [plt.Line2D([0], [1], color=color, linewidth = 0.9,linestyle='-', label=f"$M_{{Z'}} = {m_z}$") for m_z, color in colors.items()]
    # Dummy plots for rinv values
    legend_handles_rinv = [plt.Line2D([0], [1], color='black',linewidth = 0.9, linestyle=linestyle, label=f"$r_{{inv}} = {rinv}$") for rinv, linestyle in line_styles.items()]

    # Plot the  ROC curves

    plt.axhline(y=1, color='gray', linestyle='--')

    # Create the first legend for MZ values
    first_legend = plt.legend(handles=legend_handles_mz, loc=[0.735,0.3])
    ax = plt.gca().add_artist(first_legend)

    plt.legend(handles=legend_handles_rinv, loc=[0.75,0.05])
    plt.grid()

    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")

    plt.savefig(output_dir)
