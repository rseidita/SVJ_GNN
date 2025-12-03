# SVJ GNN

This package eases the processing of hadronic jets stored in NTuples into graphs and train GNNs of any flavor. Spacial (à la ParticleNET) and Lund graph (usually better) representation of hadronic jets are supported.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)

## Installation

If you are working on `lxplus` at CERN, you can use a compatible LCG envirionment. The one I used is:
```
source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh
```

Otherwise, you can prepare your venv (exchange all instances of `mamba` with `conda` if that is your jam):
```
mamba create -n gnn python=3.11
mamba activate gnn

mamba install numpy scipy matplotlib tqdm

mamba install numba

# If you run on CPU
mamba install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 cpuonly -c pytorch

# If you have a CUDA-enabled GPU
# mamba install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=11.8 -c pytorch -c nvidia

mamba install pyg -c pyg
mamba install vector uproot awkward networkx

# If you want to use MDMM
export OLD_PWD=$PWD
python -m pip install mdmm && cd $(python -m site | grep "site-packages'," | cut -d"'" -f2)/mdmm && git clone git@github.com:fleble/mdmm.git && cp mdmm/mdmm/mdmm.py . && rm -rf mdmm
cd $OLD_PWD
```

Clone this repo:
```
git clone git@github.com:rseidita/SVJ_GNN.git
```

## Usage

A complete example is given in the aptly named "example" notebook. Data loading and processing, model definition and training are covered.

## Contact

roberto.seidita@cern.ch
