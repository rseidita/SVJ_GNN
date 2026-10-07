#!/bin/bash
# HTCondor worker wrapper for the GAE pipeline.
#
#   run_job.sh <stage> <sample> <shard> [extra arguments...]
#
# stage is one of: produce, score, train, contamination; sample names a sample in
# gae_config.json (ignored by train, which takes --train-on instead).
# The remaining arguments are passed straight to the stage's python script.
# Set GAE_OUTPUT_EOS to copy the stage output to EOS with xrdcp.

set -e

STAGE=$1
SAMPLE=$2
SHARD=$3
shift 3

echo "Running stage '${STAGE}' sample '${SAMPLE}' shard '${SHARD}' on $(hostname)"

tar -xzf SVJ_GNN.tar.gz
cd SVJ_GNN

source setup_gae_env.sh

case ${STAGE} in
    produce)
        python produce_graphs.py --samples "${SAMPLE}" --shard "${SHARD}" "$@"
        OUTPUTS="data"
        ;;
    score)
        python score_gae.py --samples "${SAMPLE}" --shard "${SHARD}" "$@"
        OUTPUTS="output_scores"
        ;;
    train)
        python train_gae.py "$@"
        OUTPUTS="training"
        ;;
    contamination)
        YEAR="${SAMPLE%%__*}"
        BUCKET="${SAMPLE#*__}"
        mkdir -p contamination_shards
        python signal_contamination.py \
            --years "${YEAR}" \
            --bucket "${BUCKET}" \
            --shard "${SHARD}" \
            --output-json "contamination_shards/${SAMPLE}_${SHARD}.json" \
            "$@"
        OUTPUTS="contamination_shards"
        ;;
    *)
        echo "Unknown stage '${STAGE}'; expected produce, score, train or contamination"
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
    tar -czf "../${STAGE}_${SAMPLE}_${SHARD}_output.tar.gz" "${OUTPUTS}"
fi

echo "Stage '${STAGE}' sample '${SAMPLE}' shard '${SHARD}' finished"
