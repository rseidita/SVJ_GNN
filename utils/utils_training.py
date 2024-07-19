from utils.JetGraphProducer import JetGraphProducer
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
import uproot
import pandas as pd



def Normalize(base_directory_bkg, base_directory_signal, model_type):
    """Create means and stds for the normalization.

    Args:
        base_directory_bkg(str): Directory of the backgrounds
        base_directory_signal(str): Directory of the signal
        model_type (str): Type of the model: LN3, LN3p, LN5, LN5+

    Returns: means, stds (np.array)
        
    """


    bkg_directory = [
        base_directory_bkg + "QCD_new/QCD_HT300to500",
        base_directory_bkg + "QCD_new/QCD_HT500to700",
        base_directory_bkg + "QCD_new/QCD_HT700to1000", 
        base_directory_bkg + "QCD_new/QCD_HT1000to1500", 
        base_directory_bkg + "QCD_new/QCD_HT1500to2000", 
        base_directory_bkg + "QCD_new/QCD_HT2000toInf",
        
        base_directory_bkg + "TTJet",
        ]  
    
    Bkg = {
        f"QCD_HT300to500_{model_type}": None,
        f"QCD_HT500to700_{model_type}": None,
        f"QCD_HT700to1000_{model_type}" : None,
        f"QCD_HT1000to1500_{model_type}" : None,
        f"QCD_HT1500to2000_{model_type}" : None,
        f"QCD_HT2000toInf_{model_type}" : None,

        f"TTJet_{model_type}" : None,}

    signal_directory = [
        base_directory_signal + "SVJ_mMed-700GeV_mDark-20GeV_rinv-0.3_alpha-peak",
        base_directory_signal + "SVJ_mMed-700GeV_mDark-20GeV_rinv-0.5_alpha-peak",
        base_directory_signal + "SVJ_mMed-700GeV_mDark-20GeV_rinv-0.7_alpha-peak",

        base_directory_signal + "SVJ_mMed-800GeV_mDark-20GeV_rinv-0.3_alpha-peak",
        base_directory_signal + "SVJ_mMed-800GeV_mDark-20GeV_rinv-0.5_alpha-peak",
        base_directory_signal + "SVJ_mMed-800GeV_mDark-20GeV_rinv-0.7_alpha-peak",

        base_directory_signal + "SVJ_mMed-900GeV_mDark-20GeV_rinv-0.3_alpha-peak",
        base_directory_signal + "SVJ_mMed-900GeV_mDark-20GeV_rinv-0.5_alpha-peak",
        base_directory_signal + "SVJ_mMed-900GeV_mDark-20GeV_rinv-0.7_alpha-peak",

        base_directory_signal + "SVJ_mMed-1000GeV_mDark-20GeV_rinv-0.3_alpha-peak",
        base_directory_signal + "SVJ_mMed-1000GeV_mDark-20GeV_rinv-0.5_alpha-peak",
        base_directory_signal + "SVJ_mMed-1000GeV_mDark-20GeV_rinv-0.7_alpha-peak",

        base_directory_signal + "SVJ_mMed-1100GeV_mDark-20GeV_rinv-0.3_alpha-peak", 
        base_directory_signal + "SVJ_mMed-1100GeV_mDark-20GeV_rinv-0.5_alpha-peak", 
        base_directory_signal + "SVJ_mMed-1100GeV_mDark-20GeV_rinv-0.7_alpha-peak", 
         
        base_directory_signal + "SVJ_mMed-1200GeV_mDark-20GeV_rinv-0.3_alpha-peak", 
        base_directory_signal + "SVJ_mMed-1200GeV_mDark-20GeV_rinv-0.5_alpha-peak", 
        base_directory_signal + "SVJ_mMed-1200GeV_mDark-20GeV_rinv-0.7_alpha-peak", 
         
        base_directory_signal + "SVJ_mMed-1300GeV_mDark-20GeV_rinv-0.3_alpha-peak", 
        base_directory_signal + "SVJ_mMed-1300GeV_mDark-20GeV_rinv-0.5_alpha-peak",
        base_directory_signal + "SVJ_mMed-1300GeV_mDark-20GeV_rinv-0.7_alpha-peak", 

        base_directory_signal + "SVJ_mMed-1400GeV_mDark-20GeV_rinv-0.3_alpha-peak",
        base_directory_signal + "SVJ_mMed-1400GeV_mDark-20GeV_rinv-0.5_alpha-peak",
        base_directory_signal + "SVJ_mMed-1400GeV_mDark-20GeV_rinv-0.7_alpha-peak",

        base_directory_signal + "SVJ_mMed-1500GeV_mDark-20GeV_rinv-0.3_alpha-peak",
        base_directory_signal + "SVJ_mMed-1500GeV_mDark-20GeV_rinv-0.5_alpha-peak",
        base_directory_signal + "SVJ_mMed-1500GeV_mDark-20GeV_rinv-0.7_alpha-peak",

        base_directory_signal + "SVJ_mMed-3000GeV_mDark-20GeV_rinv-0.3_alpha-peak",
        base_directory_signal + "SVJ_mMed-3000GeV_mDark-20GeV_rinv-0.5_alpha-peak",
        base_directory_signal + "SVJ_mMed-3000GeV_mDark-20GeV_rinv-0.7_alpha-peak"
        ]

    signal= {
        f"SVJ_700_0p3_{model_type}" : None, f"SVJ_700_0p5_{model_type}" : None, f"SVJ_700_0p7_{model_type}" : None, 
        f"SVJ_800_0p3_{model_type}" : None, f"SVJ_800_0p5_{model_type}" : None, f"SVJ_800_0p7_{model_type}" : None,
        f"SVJ_900_0p3_{model_type}" : None, f"SVJ_900_0p5_{model_type}" : None, f"SVJ_900_0p7_{model_type}" : None,
        f"SVJ_1000_0p3_{model_type}" : None, f"SVJ_1000_0p5_{model_type}" : None, f"SVJ_1000_0p7_{model_type}" : None,
        f"SVJ_1100_0p3_{model_type}" : None, f"SVJ_1100_0p5_{model_type}" : None, f"SVJ_1100_0p7_{model_type}" : None,
        f"SVJ_1200_0p3_{model_type}" : None, f"SVJ_1200_0p5_{model_type}" : None, f"SVJ_1200_0p7_{model_type}" : None,
        f"SVJ_1300_0p3_{model_type}" : None, f"SVJ_1300_0p5_{model_type}" : None, f"SVJ_1300_0p7_{model_type}" : None,
        f"SVJ_1400_0p3_{model_type}" : None, f"SVJ_1400_0p5_{model_type}" : None, f"SVJ_1400_0p7_{model_type}" : None,
        f"SVJ_1500_0p3_{model_type}" : None, f"SVJ_1500_0p5_{model_type}" : None, f"SVJ_1500_0p7_{model_type}" : None,
        f"SVJ_3000_0p3_{model_type}" : None, f"SVJ_3000_0p5_{model_type}" : None, f"SVJ_3000_0p7_{model_type}" : None,
        }

    for directory, key in zip(bkg_directory, Bkg):
        Bkg[key] = JetGraphProducer(directory,key)

    for directory, key in zip(signal_directory, signal):
        signal[key] = JetGraphProducer(directory,key)


    means, stds = 0., 0.
    weights = []

    for key in Bkg:
        for graph in Bkg[key]:
            means += graph.x.sum(dim=0)*graph.w
            weights.append(graph.w)
    for key in signal:
        for graph in signal[key]:
            means += graph.x.sum(dim=0)*graph.w
            weights.append(graph.w)
    means /= sum(weights)

    for key in Bkg:
        for graph in Bkg[key]:   
            stds += ((graph.x - means)**2).sum(dim=0)*graph.w
    for key in signal:
        for graph in signal[key]:   
            stds += ((graph.x - means)**2).sum(dim=0)*graph.w

    stds /= sum(weights)
    stds = torch.sqrt(stds)

    return means, stds

def get_model_parameters(model_type):
    """Get model(graph) parameters.

    Args:
        model_type: Name of model: LN3, LN3p, LN3pp, LN5, LN5p, LN5pp (p: save energy fractions, pp:save energyfrac and four vectors)

    Returns: nLund_variables(int), energy_fractions(bool), four_vectors(bool)
        
    """
    if model_type == "LN3":
        energy_fractions = False
        four_vectors = False
        nLund_variables = 3
    if model_type == "LN3p":
        energy_fractions = True
        four_vectors = False
        nLund_variables = 3
    if model_type == "LN3pp":
        energy_fractions = True
        four_vectors = True
        nLund_variables = 3
    if model_type == "LN5":
        energy_fractions = False
        four_vectors = False
        nLund_variables = 5
    if model_type == "LN5p":
        energy_fractions = True
        four_vectors = False
        nLund_variables = 5
    if model_type == "LN5pp":
        energy_fractions = True
        four_vectors = True
        nLund_variables = 4
    return nLund_variables, energy_fractions, four_vectors



def output_directory(name):
    """Create output directory with the name of the training and
        return the path s.t. I can save everything in it.

    Args:
        name(str): Name of the training

    Returns: path to output directory (str)
        
    """
    output_path =  os.path.join(os.getcwd(),"output",name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print(f"==> Created directory {output_path}")
    
    return output_path

def save_loss(training_losses, validation_losses, name):
    """Save Losses in root file and make plots of it.

    Args:
        training_losses(np.array): array of training losses
        validation_losses(np.array): array of validation losses


    Returns: 
        
    """

    output_dict = {
        "Training_loss": np.array(training_losses),
        "Validation_loss": np.array(validation_losses),
    }

    output_path = output_directory(name)

    #take output and convert each column to a list and safe it in a root file
    output_file_path = output_path + f"/{name}_loss.root"
    file = uproot.recreate(output_file_path)
    df = pd.DataFrame(output_dict,columns=output_dict.keys())
    file["Output"] = df
    file.close()
    print(f"Output save to {output_file_path}")

    #make also the loss plot
    # Iterate over each key in roc dictionary and plot the curves
    
    x_data = range(0,len(training_losses))
    plt.plot(x_data, training_losses, color = '#457B9D', linewidth = 0.9, linestyle = "-", label="Training Loss")
    plt.plot(x_data, validation_losses,color = '#457B9D', linewidth = 0.9, linestyle = ":", label="Validation Loss")

    plt.xlabel("Epoch")
    plt.ylabel("BinaryCrossEntropy")
    plt.yscale("log")
    plt.legend()

    output_path = output_directory(name)
    outout_plot_name = name + "_loss.png"
    plt.savefig(os.path.join(output_path,outout_plot_name), dpi = 200)
    plt.close()
    print(f"{outout_plot_name} has been saved.")

 


def generate_name_with_versionnumber(base_name, directory):
    """Generate the name with a version number so that I dont' overwrite stuff.

    Args:
        base_name(str): Name of the experiment
        directory(str): current working directory

    Returns: unique_name(str): base_name with version number
        
    """
        
    version = 1
    unique_name = f"{base_name}_vs{version}"

    while os.path.exists(os.path.join(directory, "output", unique_name)):
        version += 1
        unique_name = f"{base_name}_vs{version}"

    return unique_name



class early_stop:
    """Early stops the training if validation loss does not improve """
    def __init__(self, patience=50, naverage = 5):
        self.patience = patience
        self.naverage = naverage
        self.counter = 0
        self.best_loss = 999 # start somewhere large
        self.early_stop = False
        self.best_epoch = 0
        self.loss_dict = {f"{i}": None for i in range(0, 10)} # create dictionary s.t. I can put the last 10 validation losses in it
        self.avg_loss = 999
    
    def __call__(self, validation_loss, model, epoch, output_path, name):
        if epoch == 0:
            self.best_loss = validation_loss
        self.calculate_avg_val_loss(validation_loss,epoch)
        self.early_stopping(epoch, model, output_path, name)

    def calculate_avg_val_loss(self, validation_loss, epoch):
        self.loss_dict[f"{epoch % self.naverage}"] = validation_loss
        length = len({k: v for k, v in self.loss_dict.items() if v is not None})
        self.avg_loss = sum(filter(None,self.loss_dict.values()))/length

    # check whether early stopping is necessary
    def early_stopping(self, epoch, model, output_path, name):
        if self.avg_loss > self.best_loss:
            self.counter += 1
            print(f"Avergae loss = {self.avg_loss} is larger than {self.best_loss}")
            if self.counter >= self.patience:
                self.early_stop = True
                print(f"==> Early stop because of overfitting")
                print(f"Model saved for epoch {self.best_epoch}")
        else:
            self.best_loss = self.avg_loss
            self.counter = 0
            self.best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                }, os.path.join(output_path, name + ".pt"))



    # # always save the model with the lowest validation loss so far
    # def save_checkpoint(self, model, epoch, output_path, name):
    #     if self.avg_loss <= self.best_loss:
    #         self.best_epoch = epoch
    #         torch.save({
    #             'epoch': epoch,
    #             'model_state_dict': model.state_dict(),
    #             }, os.path.join(output_path, name + ".pt"))


