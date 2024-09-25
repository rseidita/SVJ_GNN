# SVJ GNN

This package facilitates the processing of hadronic jets stored in NTuples into graphs and allows training Graph Neural Networks (GNNs) of any flavor. Both Spacial (à la ParticleNET) and Lund graph representations (typically more effective) of hadronic jets are supported.

## Table of Contents

- [Model and Input Graphs](#model-and-input-graphs)
- [Training](#training)
- [Evaluation](#evaluation)
- [Jet Tagger](#jet-tagger)
- [Determination of the Working Point (WP)](#determination-of-the-working-point-wp)

## Model and Input Graphs

The model used for training is specified in `architectures/LundNet.py` and can be modified as needed.

Input graphs can be produced using a Jupyter script like `Scripts/ProduceGraphs.ipynb`, which must be run locally. The script utilizes the `JetGraphProducer`, providing several options to adjust the graphs.

## Training

Model training is handled by the `TrainLundNet.py` script. It can be run on a GPU using the command `sbatch runTraining.sh`. The following parameters can be provided:

- `--epochs`: Number of epochs for training  
  *type: int, default: 5*

- `--batch`: Batch size  
  *type: int, default: 64*

- `--name`: Name of the training session to save the losses and model  
  *type: str, default: "LundNet"*

- `--model`: Model to train: `LN3`, `LN3p`, `LN3pp`, `LN5`, `LN5p`, `LN5pp` (where `p` saves energy fractions and `pp` saves energy fractions and four vectors)  
  *type: str, default: "LN3"*

- `--lr`: Learning rate of the model  
  *type: float, default: 0.001*

- `--jet_idx`: Jet to focus on: `0=leading jet`, `1=subleading jet`, `2=subsubleading jet`, `None=all graphs in the dataset`  
  *type: int, default: None*

- `--eval_type`: Type of PT file to evaluate on: `_3J` (each jet single)  
  *type: str, default: ""*

Early stopping is implemented in `utils/utils_training.py`. Once the training is completed, the model is automatically evaluated.

After training and evaluation, the output is saved in a directory specified by the `--name` option. This directory, located in the `output` directory, contains the model, normalization data, losses in a root file, and various evaluation plots.

## Evaluation

To evaluate the model in more detail, use the `Evaluation.py` script. This script also allows you to evaluate the model on the training or validation set and produce additional plots. Run the evaluation with `sbatch runEvaluation.sh`.

## Jet Tagger

The `JetTagger.py` script can be used to recreate input TFiles with an additional branch called `FatJet_SVJtagger`. The current implementation works but is slow and requires optimization. The approach is to add the corresponding event number to each graph during their production. Then, all graphs are loaded through the network, and the event number and whether it is an SVJ are saved in a dictionary. Next, all root files from which the graphs were produced are loaded. The event numbers for the graphs are matched to the files, and the additional branch is added. The output files are saved to the specified output directory.

`JetTagger_vs2` uses the same principle but works for non-unique event numbers over all events.

To create TFiles with only histograms of MT containing either 0, 1, or 2 SVJ, use the `ProduceMTDistribution.py` script. This script processes the graphs (with MT added during production), runs them through the network, and categorizes them into different histograms depending on the number of SVJ in an event. The histograms are saved in the `MtDistributions` directory.

## Determination of the Working Point (WP)

The working point, determining after which output score a jet is marked as semi-visible, can be determined using several methods. Three methods are implemented:

1. **Choose it by hand**
2. **Highest SIC on event level, averaged over all signal hypotheses**
3. **Highest SIC on jet level, averaged over all signal hypotheses**

In `ProduceMTDistribution.py`, specify the WP to use. To use SIC on the event level, run the `SIC.py` script beforehand to create the ROC curve and SIC on the event level. The ROC, SIC, etc., are saved in a `.pkl` file named `EventLevel_ROC.pkl` in the corresponding output directory.

For jet-level SIC, the `_eval_ROC_AUC.pkl` file is used, which is produced during the model evaluation.
