from comet_ml import Experiment

from utils.LundTreeUtilities import OnTheFlyNormalizer
from data_loading.graphDataLoader import GraphDataLoader
from architectures.LundNet import LundNet
from tqdm import tqdm 

import time
import torch
import argparse
import os
import pickle
import numpy as np

import utils.utils_evaluation as ueval
import utils.utils_training as utrain

import mdmm


def main(epochs, batch_size, learning_rate, name, model_type, eval_type, jet_idx):
    """Train the model and evaluate it.

    Args:
        epochs(int): number of epochs to train for
        batch_size(int): size of each batch
        learning_rate(float):
        name(str): name of the training
        model_type (str): Type of the model: LN3, LN3p, LN5, LN5+
        jet_idx(int): Which jet to look at (0,1,2), default = None -> look at all graphs in the window

    Returns: -
        
    """

    start = time.time()

    ###--------- Prepare Model, Paths to input graphs  -----------------------------------------------------------

    # create output directory and get path to it
    output_path = utrain.output_directory(name)

    # Directory for the graphs
    base_directory_bkg = "/pnfs/psi.ch/cms/trivcat/store/user/rhampp/forTraining/Preselection_applied/Bkg_sorted/"
    base_directory_SVJ = "/pnfs/psi.ch/cms/trivcat/store/user/rhampp/forTraining/Preselection_applied/SVJ_26_08/"

    # create on the fly normalizer and save the mean and std 
    means, stds = utrain.Normalize(base_directory_bkg, base_directory_SVJ, model_type)
    normalizer = OnTheFlyNormalizer(["x"], means, stds)
    normalize_dict = {
        'means': means,
        'stds': stds}
    
    with open(os.path.join(output_path, "Normalization.pkl"), 'wb') as f:
        pickle.dump(normalize_dict, f)
    print(f"Normalization saved in {os.path.join(output_path, 'Normalization.pkl')}")

    # List: [paths to the "processed" directory containting the .pt files with the graph, name of the .pt file]
    datasets = [
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

        [base_directory_SVJ + "SVJ_mMed-3000GeV_mDark-20GeV_rinv-0.3_alpha-peak", f"SVJ_3000_0p3_{model_type}"],
        [base_directory_SVJ + "SVJ_mMed-3000GeV_mDark-20GeV_rinv-0.5_alpha-peak", f"SVJ_3000_0p5_{model_type}"],
        [base_directory_SVJ + "SVJ_mMed-3000GeV_mDark-20GeV_rinv-0.7_alpha-peak", f"SVJ_3000_0p7_{model_type}"],
    ]

    ###--------- Prepare GraphDataLoader  -----------------------------------------------------------
    ## Specify the datasetfractions to train on and then create the dataloader for signal and background for training and validation
    
    ## Datasetfractions to train on
    # Background: 25% QCD, 12.5% TTJet, 12.5% EW ( 6.25%Z, 6.25%W)
    # dfqcd = 0.25/6
    # dfttjet = 0.25/2
    # dfew = 0.25/4
    # dfZ = dfew/7
    # dfW = dfew

    # Background: 16.7%% QCD, 16.7% TTJet, 16.7% EW ( 8.3%Z, 8.3%W)
    dfbkg = 0.5/3
    dfqcd = dfbkg/6
    dfttjet = dfbkg
    dfew = dfbkg/2
    dfZ = dfew/7
    dfW = dfew


    # Signal: each hypothesis is 1.6% , in total 50% signal
    dfs = 0.5/30
    df = [dfqcd]*6 + [dfttjet] + [dfW] + [dfZ]*7 + [dfs]*30 # Train on 50% background, and 50 % signal

    # Create the data loaders for training and validation
    dataloader_training = GraphDataLoader(
        datasets,
        window_to_load=[[0., 0.6] for _ in range(len(datasets))],
        dataset_fraction=df,
        batch_size=batch_size,
        normalizer=normalizer,
    )
    
    dataloader_validation = GraphDataLoader(
        datasets,
        window_to_load=[[0.6, 0.7] for _ in range(len(datasets))],
        dataset_fraction=df,
        batch_size=batch_size,
        normalizer=normalizer,
    )
    
    ###--------- Create the Model -----------------------------------------------------------

    # Input parameters for the Model
    nLund_variables, energy_fractions, four_vectors = utrain.get_model_parameters(model_type)
    if energy_fractions:
        nLund_variables += 4
 
    # Create the model
    model = LundNet(
        conv_params = [[128, 128], [128, 128], [64, 64], [64, 64], [32, 32], [32, 32]],
        fc_params = [(32, 0.8)],
        input_dims = nLund_variables,
        add_fractions_to_lund = four_vectors
    )

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Load model on gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Define the loss
    loss_criterion = torch.nn.BCELoss()
    


    ###--------- Train the network -----------------------------------------------------------

    training_losses, validation_losses = train_and_validate(model, epochs, dataloader_training, dataloader_validation, loss_criterion, device, optimizer, name)
    print(f"==> Training completed in {time.time()-start:.2f} seconds")

    print("==> Saving losses...")
    utrain.save_loss(training_losses, validation_losses, name)
    
    print("Traininig finished")

     ###--------- Evaluate the model -----------------------------------------------------------

    print("==> Evaluate Model...")
    start_eva = time.time()

    # Run test dataset through model and save output in a dict 
    evaluation_dict = ueval.Evaluate_model(base_directory_bkg, base_directory_SVJ,model_type,eval_type, batch_size, normalizer, model, device, window_to_load=[0.7,1], jet_idx=None)

    # Compute roc and auc, safe them also in pickels
    roc, auc = ueval.roc_auc(evaluation_dict, name)
    ueval.plot_auc(auc, name)
    ueval.plot_roc(roc, name)
    ueval.plot_scores(evaluation_dict,name)
    ueval.plot_sic(roc, name)
    ueval.plot_dnn(evaluation_dict, name)
    ueval.plot_sic_small(roc,name)
    ueval.plot_dnn(evaluation_dict,name)
    ueval.plot_dnn_2(evaluation_dict, name)

    print(f"==> Evaluation completed in {time.time()-start_eva:.2f} seconds")

    return

def train_and_validate(model, epochs, dataloader_training, dataloader_validation, loss_criterion, device, optimizer, name):
    """Train and validate model.

    Args:
        model: model to train
        epochs(int): number of epochs to train for
        dataloader_training(GraphDataLoader): dataloader for training
        dataloader_validation(GraphDataLoader): dataloader for validation
        loss_criterion:
        device: GPU or CPU
        optimizer: ADAM


    Returns: training_losses(List), validation_losses(List)
        
    """

    training_losses = []
    validation_losses = []


    # initalize early stopping class
    early_stop = utrain.early_stop()

    # get output_path
    output_path = utrain.output_directory(name)
    

    for epoch in range(0,epochs):
        #training
        batch_counter = 0
        loss_per_batch = 0


        for batch, data in enumerate(dataloader_training):
        # for batch, data in tqdm(enumerate(dataloader_training), total=len(dataloader_training), desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            model.train() # set into training modus for backpropagation
            data = data.to(device) # send data to vram of gpu!
            target = data.y # this is my label
            
            optimizer.zero_grad() #make sure gradients are empty
            out = model(data) # fill data into model and safe the output
            loss_ =  loss_criterion(out[:,0], target) # takes the first argument of the output(could also take the other)
            score = out[:,0].cpu()
            score = score.detach().numpy()
            loss_per_batch += loss_.item()

            loss_.backward()
            optimizer.step()
            batch_counter += 1

        # calculate loss per epoch and log it
        epoch_loss_training = loss_per_batch/batch_counter
        training_losses.append(epoch_loss_training)
        experiment.log_metric("Training loss", epoch_loss_training, epoch=epoch)

        #validation
        batch_counter = 0
        loss_per_batch = 0

        for batch, data in enumerate(dataloader_validation):
            model.eval()
            data = data.to(device)
            target = data.y

            out = model(data)
            score = out[:,0].cpu()
            score = score.detach().numpy()

            loss_ = loss_criterion(out[:,0], target)
            loss_per_batch += loss_.item()
            batch_counter += 1

        
        # calculate loss per epoch and log it
        epoch_loss_validation = loss_per_batch/batch_counter
        validation_losses.append(epoch_loss_validation)
        experiment.log_metric("Validation loss", epoch_loss_validation, epoch=epoch)

        ## check for early stop
        early_stop(epoch_loss_validation,model,epoch,output_path,name)
        if early_stop.early_stop == True:
            break
        ## log the average loss asw ell
        experiment.log_metric("Average Validation loss", early_stop.avg_loss, epoch=epoch)

    # Safe output Model
    print("==> Saving Model")      
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
        }, os.path.join(output_path, name + ".pt"))
    print("Model saved to ", output_path + name + "_end.pt")

       

    return training_losses, validation_losses



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epochs",
        default=5,
        help="Number of epochs to train on",
        type=int,
        )
    parser.add_argument(
        "--batch",
        default=64,
        help="Batch size",
        type=int,
        )
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

    parser.add_argument(
        "--lr",
        default=0.001,
        help="Learning rate of the model",
        type=float,
        )

    parser.add_argument(
        "--jet_idx",
        default=None,
        help="Which jet to look at: 0=leading jet, 1=subleading jet, 2=subsubleading jet, None = all graphs in the dataset",
        type=int,
    )

    parser.add_argument(
        "--eval_type",
        default="",
        help="Which pt file to evaluate on: _3J(each jet single)",
        type=str,
    )
    


    args = parser.parse_args()

    epochs = args.epochs
    batch_size = args.batch
    learning_rate = args.lr
    name = args.name
    model_type = args.model
    jet_idx = args.jet_idx
    eval_type = args.eval_type

    print("==> Running LundNet ")

    experiment = Experiment(
        api_key="elPFdQpSrPZ7TQEWYgdIjWqDU",
        project_name=name,
        workspace="rhampp"
        )
    
    name = utrain.generate_name_with_versionnumber(name, os.getcwd())
    utrain.output_directory(name)
    
    #log hyperparameters in the experiment
    hyper_params = {
        "learning_rate": learning_rate,
        "epochs": epochs,
        "Model" : model_type,
        "remark": "fixed early stopping",
    }

    experiment.log_parameters(hyper_params)
    experiment.set_name(name)

    
    main(epochs, batch_size, learning_rate, name, model_type, eval_type, jet_idx=None)


