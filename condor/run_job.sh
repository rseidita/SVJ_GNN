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
        ;;
    score)
        python score_gae.py --samples "${SAMPLE}" --shard "${SHARD}" "$@"
        ;;
    train)
        python train_gae.py "$@"
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

if [ -z "${OUTPUTS}" ]; then
    OUTPUTS=$(python3 -c '
import json, sys
stage = sys.argv[1]
args = sys.argv[2:]
cfg_path = "gae_config.json"
out_dir = None
for i, a in enumerate(args):
    if a == "--config" and i + 1 < len(args):
        cfg_path = args[i+1]
    elif a == "--output-dir" and i + 1 < len(args):
        out_dir = args[i+1]
if out_dir:
    print(out_dir)
    sys.exit(0)
defaults = {"produce": "data", "train": "training", "score": "output_scores"}
default_out = defaults.get(stage, "")
try:
    with open(cfg_path) as f:
        d = json.load(f)
    section = "production" if stage == "produce" else ("training" if stage == "train" else "scoring")
    val = d.get(section, {}).get("output_dir", default_out)
    print(val or default_out)
except Exception:
    print(default_out)
' "${STAGE}" "$@")
fi

if [ -n "${GAE_OUTPUT_EOS}" ]; then
    echo "Saving ${OUTPUTS} to ${GAE_OUTPUT_EOS}"
    for file in $(find "${OUTPUTS}" -type f); do
        if [[ "${GAE_OUTPUT_EOS}" == root://* ]]; then
            xrdcp -f "${file}" "${GAE_OUTPUT_EOS}/${file}"
        else
            dest="${GAE_OUTPUT_EOS}/${file}"
            mkdir -p "$(dirname "${dest}")"
            cp -f "${file}" "${dest}"
        fi
    done
else
    # No EOS target, so hand the outputs back through condor file transfer.
    tar -czf "../${STAGE}_${SAMPLE}_${SHARD}_output.tar.gz" "${OUTPUTS}"
fi

echo "Stage '${STAGE}' sample '${SAMPLE}' shard '${SHARD}' finished"
