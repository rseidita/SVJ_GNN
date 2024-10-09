import importlib.util as imp
import argparse
import os
from utils.GAETrainer import GAETrainer
#from utils.GAEEvaluator import GAEEvaluator
from utils.DataManager import TrainingDataManager
from utils.plottingHelpers import plot_losses, plot_rocs


parser = argparse.ArgumentParser()
parser.add_argument(
    "-c",
    "--configuration",
    required=True,
    type=str,
    help="Path to folder containing the configuration files",
    )
''' parser.add_argument(
    "-s",
    "--step",
    required=True,
    choices=["train", "eval", "optimize", "apply"],
    type=str,
    help="Step to be run",
    )
'''
parser.add_argument(
    "-o",
    "--output",
    required=True,
    type=str,
    help="Output path for the results",
    )
args = parser.parse_args()
    
config_path=args.configuration
output_path=args.output

def load_data_loading_config(config_path):
    '''
    Load the data loading configuration from the specified path

    Args:
    config_path (str): Path to the configuration folder

    Returns:
    config_data_loading (module): Module containing the data loading configuration
    '''

    if not os.path.exists(config_path + "/data_loading_options.py"):
        raise FileNotFoundError(f"Configuration file data_loading_options.py not found in {config_path}")

    data_loading_spec = imp.spec_from_file_location("data_loading_options", config_path + "/data_loading_options.py")
    config_data_loading = imp.module_from_spec(data_loading_spec)
    data_loading_spec.loader.exec_module(config_data_loading)

    return config_data_loading

def load_training_config(config_path):
    '''
    Load the training configuration from the specified path

    Args:
    config_path (str): Path to the configuration folder

    Returns:
    config_training_options (module): Module containing the training configuration
    '''

    if not os.path.exists(config_path + "/training_options.py"):
        raise FileNotFoundError(f"Configuration file training_options.py not found in {config_path}")

    training_spec = imp.spec_from_file_location("training_options", config_path + "/training_options.py")
    config_training_options = imp.module_from_spec(training_spec)
    training_spec.loader.exec_module(config_training_options)

    return config_training_options

def train(config_data_loading, config_training_options, output_path):

    '''
    Train the GAE model

    Args:
    config_data_loading (module): Module containing the data loading configuration
    config_training_options (module): Module containing the training configuration
    output_path (str): Path to the output folder

    Returns:
    None
    '''
    

    data_manager = TrainingDataManager(config_data_loading,config_training_options.batch_size)
    trainer = GAETrainer(data_manager, config_training_options, config_data_loading)

    training_losses, validation_losses = trainer.train_and_validate(output_path)
    result, fpr, tpr, aucx = trainer.test()


    plot_losses(training_losses, validation_losses, output_path, config_training_options.constraints_manager)
    plot_rocs(result, fpr, tpr, aucx, output_path)



config_data_loading=load_data_loading_config(config_path)

config_training_options=load_training_config(config_path)

train(config_data_loading, config_training_options, output_path)
