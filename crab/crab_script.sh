#!/bin/bash
# CRAB scriptExe entry point for the GAE pipeline.
#
# CRAB passes the job number as the first argument and lists the files it
# assigned to this job in PSet.py. Only the per-file stages belong here:
# graph production and scoring. Training is a single long GPU job, which does
# not fit the CRAB model, so use condor/submit_condor.py --stage train.
#
# Configure through the crab configuration's JobType.scriptArgs:
#   GAE_STAGE      'produce' or 'score'
#   GAE_SAMPLE     name of the sample in gae_config.json being processed
#   GAE_ARGUMENTS  extra arguments for the stage's python script

set -e

JOB_NUMBER=$1
echo "CRAB job ${JOB_NUMBER} on $(hostname)"

GAE_STAGE=${GAE_STAGE:-score}
GAE_SAMPLE=${GAE_SAMPLE:?GAE_SAMPLE must name a sample in gae_config.json}
export GAE_SAMPLE

cp PSet.py SVJ_GNN_PSet.py
tar -xzf SVJ_GNN.tar.gz
cd SVJ_GNN
cp ../SVJ_GNN_PSet.py PSet.py

source setup_gae_env.sh

# The files CRAB gave this job are passed with --files; the sample's input
# JSON stays intact so cross section weights still see the whole dataset.
JOB_FILES=$(python - <<'PYTHON'
import PSet

print(" ".join(str(name) for name in PSet.process.source.fileNames))
PYTHON
)
echo "Job files: ${JOB_FILES}"

case ${GAE_STAGE} in
    produce)
        python produce_graphs.py --samples "${GAE_SAMPLE}" \
            --files ${JOB_FILES} ${GAE_ARGUMENTS}
        OUTPUTS="data"
        ;;
    score)
        python score_gae.py --samples "${GAE_SAMPLE}" \
            --files ${JOB_FILES} ${GAE_ARGUMENTS}
        OUTPUTS="scores"
        ;;
    *)
        echo "Unknown GAE_STAGE '${GAE_STAGE}'; expected produce or score"
        exit 1
        ;;
esac

# CRAB collects the declared output files from the job's top directory.
tar -czf "../gae_${GAE_STAGE}_${JOB_NUMBER}.tar.gz" "${OUTPUTS}"

echo "CRAB job ${JOB_NUMBER} finished"
