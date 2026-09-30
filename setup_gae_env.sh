#!/bin/bash
# Environment for the GAE pipeline. Source this, do not execute it.
#
#   source setup_gae_env.sh
#
# The LCG view provides everything except comet-ml. To get comet logging,
# create the venv once on top of the view:
#
#   source setup_gae_env.sh
#   python -m venv --system-site-packages .venv-gae
#   source .venv-gae/bin/activate
#   python -m pip install -r requirements-gae.txt
#
# Later invocations of this script pick the venv up automatically. Without it
# the pipeline still runs; train_gae.py falls back to local-only logging.

GAE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "${GAE_DIR}/../init.sh"

if [ -f "${GAE_DIR}/.venv-gae/bin/activate" ]; then
    source "${GAE_DIR}/.venv-gae/bin/activate"
fi

# Scripts import as "utils.X" / "architectures.X", so run from this directory.
export PYTHONPATH="${GAE_DIR}:${PYTHONPATH}"
