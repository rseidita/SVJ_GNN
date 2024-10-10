####################################################
####################################################
# This config file holds all necessary
# info for loading, splitting, and pre-processing
# the data for the model training and evaluation.
####################################################
####################################################

### Define the input files for the backgrounds and signals

QCD2_path="/pnfs/psi.ch/cms/trivcat/store/user/aaguadob/Training002/QCD/"
TTJet2_path="/pnfs/psi.ch/cms/trivcat/store/user/aaguadob/Training002/TTJet/"
QCD_path="/pnfs/psi.ch/cms/trivcat/store/user/aaguadob/Training001/QCD_all/"
SVJ_path="/pnfs/psi.ch/cms/trivcat/store/user/aaguadob/Training001/SVJ_all/"
SVJ2_path="/pnfs/psi.ch/cms/trivcat/store/user/aaguadob/Training002/SVJ/"
TTJets_path="/pnfs/psi.ch/cms/trivcat/store/user/aaguadob/Training001/TTJets_all"

backgrounds = {
    "samples": {
        "QCD_large001": {
            "path": QCD_path,
            "class": 0,
        },
        "TTJets_large002": {
            "path": TTJet2_path,
            "class": 0,
        },
            },
    "overall_relative_fraction": 0.5,
}

signals={
    "samples":{
        f"SVJ_mMed-{mass}GeV_mDark-20GeV_rinv-{r_inv}_alpha-peak": {
            "path": SVJ_path + f"SVJ_mMed-{mass}GeV_mDark-20GeV_rinv-{r_inv}_alpha-peak/",
            "class": 1,
        } for mass in [1000, 1500, 3000, 700, 800, 900] for r_inv in [0.3, 0.5, 0.7] if not (mass == 700 and r_inv == 0.3)
    },
    "overall_relative_fraction": 0.5
}

### Define the input features to read from the samples

import numpy as np

variables=[{
    "n_store_jets": 2,
    "use_lund_decomp":True,
    "n_lund_vars": 5,
    "save_energy_fractions":True,
    "save_4vectors_in_lund_tree":True,
    "weights":"xsec",
    "verbose":True,
    "kt_cut":None,
    "add_self_connections":True
  },
]

### Define the options for splitting of the data

test_size=0.2
validation_size=0.2