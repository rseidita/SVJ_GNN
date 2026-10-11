#!/bin/bash
# Run a small end-to-end GAE test in the control region.
#
# - Two 2018 data files: produce graphs and train the autoencoder on them only.
# - QCD and signal are never trained on; they are scored after training and
#   used in plot_gae.py for loss / ROC / SIC discrimination plots.
# - Inference uses one QCD file and one file from each of a few signal points.
#
# Run from inside SVJ_GNN/ with a valid grid proxy:
#   voms-proxy-init -voms cms
#   bash run_test_pipeline.sh
#
# To re-run scoring/plots only (skip produce + train), set:
#   SKIP_PRODUCE=1 SKIP_TRAIN=1 bash run_test_pipeline.sh

set -euo pipefail

cd "$(dirname "$0")"
SVJ_GNN_DIR="$(pwd)"

CHECKPOINT="${CHECKPOINT:-${SVJ_GNN_DIR}/training/GraphAE_graphAE/checkpoint.pt}"
HISTORY="${HISTORY:-${SVJ_GNN_DIR}/training/GraphAE_graphAE/history.json}"
SCORES_DIR="${SCORES_DIR:-${SVJ_GNN_DIR}/output_scores}"

echo "=== Sourcing LCG environment ==="
set +u
source "${SVJ_GNN_DIR}/../init.sh"
set -u

echo "=== Resolving input files from inputJson/ ==="
python - <<'PY'
from utils.GaeConfig import load_config
c = load_config("gae_config.json")

data_files = c.sample("data_2018").groups()["Run2018A"][:2]
with open("/tmp/gae_data_files.txt", "w") as f:
    f.write(" ".join(data_files) + "\n")

qcd_file = c.sample("qcd_2018").groups()["QCD_HT2000toInf"][0]
with open("/tmp/gae_qcd_files.txt", "w") as f:
    f.write(qcd_file + "\n")

signal_points = [
    "s-channel_mMed-700_mDark-20_rinv-0.5",
    "s-channel_mMed-1000_mDark-20_rinv-0.3",
    "s-channel_mMed-1500_mDark-20_rinv-0.3",
]
for point in signal_points:
    sample_name = f"signal_2018_{point}"
    files = c.sample(sample_name).files()
    with open(f"/tmp/gae_sig_{point}.txt", "w") as f:
        f.write(files[0] + "\n")

print(f"Data (train):  {len(data_files)} files")
for path in data_files:
    print(f"  {path}")
print(f"QCD (infer):   {qcd_file}")
print(f"Signal points: {len(signal_points)} (infer only)")
PY

if [[ "${SKIP_PRODUCE:-0}" != "1" ]]; then
    echo ""
    echo "=== Step 1: produce graphs for data_2018 (training sample) ==="
    rm -rf "${SVJ_GNN_DIR}/data/data_2018"
    python produce_graphs.py \
        --samples data_2018 \
        --files $(cat /tmp/gae_data_files.txt) \
        --num-workers 1
fi

if [[ "${SKIP_TRAIN:-0}" != "1" ]]; then
    echo ""
    echo "=== Step 2: train the autoencoder on the data control region for 5 epochs ==="
    rm -rf "${SVJ_GNN_DIR}/training"
    python train_gae.py \
        --train-on data_2018 \
        --epochs 5 \
        --no-comet
fi

if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "ERROR: checkpoint not found at ${CHECKPOINT}" >&2
    echo "Run without SKIP_TRAIN=1, or set CHECKPOINT=... to an existing file." >&2
    exit 1
fi

echo ""
echo "Checkpoint: ${CHECKPOINT}"
echo "History:    ${HISTORY}"

rm -rf "${SCORES_DIR}"
mkdir -p "${SCORES_DIR}"

echo ""
echo "=== Step 3: score data_2018 (inference / reference, same 2 files as training) ==="
python score_gae.py \
    --checkpoint "${CHECKPOINT}" \
    --output-dir "${SCORES_DIR}" \
    --samples data_2018 \
    --files $(cat /tmp/gae_data_files.txt)

echo ""
echo "=== Step 4: score qcd_2018 (inference only, not used in training) ==="
python score_gae.py \
    --checkpoint "${CHECKPOINT}" \
    --output-dir "${SCORES_DIR}" \
    --samples qcd_2018 \
    --files $(cat /tmp/gae_qcd_files.txt)

echo ""
echo "=== Step 5: score individual signal points (one file each) ==="
for point_file in /tmp/gae_sig_s-channel_*.txt; do
    point=$(basename "${point_file}" .txt | sed 's/gae_sig_//')
    sample_name="signal_2018_${point}"
    echo "Scoring ${sample_name} ..."
    python score_gae.py \
        --checkpoint "${CHECKPOINT}" \
        --output-dir "${SCORES_DIR}" \
        --samples "${sample_name}" \
        --files "$(cat "${point_file}")"
done

echo ""
echo "=== Step 6: produce plots ==="
rm -rf "${SVJ_GNN_DIR}/plots"
python plot_gae.py \
    --scores-dir "${SCORES_DIR}" \
    --history "${HISTORY}" \
    --checkpoint "${CHECKPOINT}" \
    --residual-sample data_2018

echo ""
echo "=== Test pipeline complete ==="
echo "Scores:  ${SCORES_DIR}/"
echo "Plots:   ${SVJ_GNN_DIR}/plots/"
echo ""
echo "Scored files:"
ls -1 "${SCORES_DIR}/"
