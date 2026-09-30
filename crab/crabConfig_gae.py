"""CRAB configuration for the per-file GAE stages.

Used for graph production and scoring, which are naturally one job per input
file. Training is a single long GPU job and does not fit the CRAB model; use
condor/submit_condor.py --stage train for that instead.

Before submitting, make the job tarball and fill in the TODO fields below:

    cd SVJ_GNN
    tar -czf crab/SVJ_GNN.tar.gz --exclude .git --exclude data \
        --exclude scores --exclude plots --exclude training \
        --exclude .venv-gae --exclude '*.root' -C .. SVJ_GNN
    crab submit -c crab/crabConfig_gae.py

The input files come from the same text file lists the rest of the pipeline
uses, through Data.userInputFiles, so no DAS dataset is needed.
"""

import os
import sys

from CRABClient.UserUtilities import config

SVJ_GNN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SVJ_GNN_DIR)

from utils.FileLists import read_file_list  # noqa: E402
from utils.GaeConfig import load_config  # noqa: E402


# Which sample and stage this submission processes.
GAE_SAMPLE = "background_example"
GAE_STAGE = "score"
GAE_ARGUMENTS = "--checkpoint checkpoint.pt"

pipeline = load_config(os.path.join(SVJ_GNN_DIR, "gae_config.json"))
sample = pipeline.sample(GAE_SAMPLE)
input_files = read_file_list(
    os.path.join(SVJ_GNN_DIR, sample.file_list)
)

config = config()

config.General.requestName = f"gae_{GAE_STAGE}_{GAE_SAMPLE}"
config.General.workArea = "crab_projects"
config.General.transferOutputs = True
config.General.transferLogs = True

config.JobType.pluginName = "Analysis"
config.JobType.psetName = os.path.join(SVJ_GNN_DIR, "crab", "PSet.py")
config.JobType.scriptExe = os.path.join(SVJ_GNN_DIR, "crab", "crab_script.sh")
config.JobType.scriptArgs = [
    f"GAE_STAGE={GAE_STAGE}",
    f"GAE_SAMPLE={GAE_SAMPLE}",
    f"GAE_ARGUMENTS={GAE_ARGUMENTS}",
]
config.JobType.inputFiles = [
    os.path.join(SVJ_GNN_DIR, "crab", "SVJ_GNN.tar.gz"),
]
config.JobType.outputFiles = [f"gae_{GAE_STAGE}_1.tar.gz"]
config.JobType.sendPythonFolder = True
config.JobType.maxMemoryMB = 4000
config.JobType.numCores = 1

# Reading the files over xrootd rather than through CMSSW means the job can
# run anywhere the files are reachable.
config.Data.userInputFiles = input_files
config.Data.splitting = "FileBased"
config.Data.unitsPerJob = 1
config.Data.totalUnits = len(input_files)
config.Data.publication = False
config.Data.outputPrimaryDataset = "SVJGAE"
config.Data.outputDatasetTag = f"gae_{GAE_STAGE}_{GAE_SAMPLE}"
# TODO: set to your own EOS area.
config.Data.outLFNDirBase = "/store/user/USERNAME/SVJGAE"

config.Site.storageSite = "T3_US_FNALLPC"
# TODO: restrict or widen as needed; the job only needs xrootd access.
config.Site.whitelist = ["T3_US_FNALLPC"]
