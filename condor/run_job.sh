#!/bin/bash
# HTCondor worker wrapper for the GAE pipeline.
#
#   run_job.sh <stage> <shard> [extra arguments...]
#
# stage is one of: produce, score, train.
# The remaining arguments are passed straight to the stage's python script.
# Set GAE_OUTPUT_EOS to copy the stage output to EOS with xrdcp.

set -e

STAGE=$1
SHARD=$2
shift 2

echo "Running stage '${STAGE}' shard '${SHARD}' on $(hostname)"

tar -xzf SVJ_GNN.tar.gz
cd SVJ_GNN

source setup_gae_env.sh

case ${STAGE} in
    produce)
        python produce_graphs.py --shard "${SHARD}" "$@"
        OUTPUTS="data"
        ;;
    score)
        python score_gae.py --shard "${SHARD}" "$@"
        OUTPUTS="scores"
        ;;
    train)
        python train_gae.py "$@"
        OUTPUTS="training"
        ;;
    *)
        echo "Unknown stage '${STAGE}'; expected produce, score or train"
        exit 1
        ;;
esac

if [ -n "${GAE_OUTPUT_EOS}" ]; then
    echo "Copying ${OUTPUTS} to ${GAE_OUTPUT_EOS}"
    for file in $(find "${OUTPUTS}" -type f); do
        xrdcp -f "${file}" "${GAE_OUTPUT_EOS}/${file}"
    done
else
    # No EOS target, so hand the outputs back through condor file transfer.
    tar -czf "../${STAGE}_${SHARD}_output.tar.gz" "${OUTPUTS}"
fi

echo "Stage '${STAGE}' shard '${SHARD}' finished"
