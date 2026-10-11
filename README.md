# SVJ GNN

SVJ GNN converts jets stored in PFNanoAOD ROOT files into
[PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) graphs. It
supports two graph representations:

- **Lund trees**: jet constituents are reclustered with the
  Cambridge–Aachen algorithm and represented as a bidirectional binary tree.
- **Spatial graphs**: constituents become nodes and nearby constituents are
  connected according to their distance in η–φ.

Processed graphs are written to chunked datasets, loaded lazily, and can be
produced in parallel across CPU processes. The repository also includes graph
autoencoder and LundNet architectures, weighted data loading, normalization,
plotting, and ROOT score production.

## Features

- PFNanoAOD constituent extraction with configurable jet and constituent limits.
- Lund variables, pseudojet four-vectors, particle energy fractions, and
  optional displacement features.
- Optional spatial edge attributes: pair mass, Cambridge–Aachen distance, and
  kT distance.
- Deterministic multiprocessing with bounded prefetch and parent-only writes.
- Chunked on-disk storage with a small in-memory LRU cache.
- Configuration-aware cache validation to prevent incompatible datasets from
  being reused silently.
- Lazy normalization and weighted sampling through `GraphDataLoader`.
- Dependency-free regression tests using Python's `unittest`.

Only **PFNanoAOD** input is currently supported. The obsolete TreeMaker input
path has been removed.

## Installation

Python 3.11 or 3.12 is recommended. Create an isolated environment from the
repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The requirements use compatible version ranges. The current test environment
uses Python 3.12, PyTorch 2.4, PyG 2.6, NumPy 1.26, Numba 0.59, Uproot 5,
Awkward 2, and Vector 1.8.

### GPU-enabled PyTorch

The PyTorch package selected by pip may not match the CUDA version on your
machine. For GPU training, install the appropriate PyTorch build using the
[official PyTorch installer](https://pytorch.org/get-started/locally/) first,
then install the remaining requirements:

```bash
python -m pip install -r requirements.txt
```

Pip will retain an already-compatible PyTorch installation.

## Expected input

`JetGraphProducer` processes every `.root` file directly inside its `root`
directory. Files must contain an `Events` tree with the usual PFNanoAOD
branches:

```text
PFCands_{pt,eta,phi,mass,pdgId}
FatJetPFCands_{jetIdx,pFCandsIdx}
nFatJet
FatJet_{pt,eta,phi}
```

When `use_displacement=True`, `PFCands_d0` and `PFCands_dz` are also required.
When `weights="xsec"`, files must additionally provide:

```text
Metadata/GenCrossSection
CutFlow/Initial
```

## Producing graphs

### Lund trees

```python
from utils.JetGraphProducer import JetGraphProducer

dataset = JetGraphProducer(
    root="data/",
    output_dataset_name="qcd_lund",
    n_store_jets=2,
    n_store_cands=None,
    use_lund_decomp=True,
    n_lund_vars=5,
    save_4vectors_in_lund_tree=True,
    save_energy_fractions=True,
    weights="xsec",
    label=0,
    graphs_per_chunk=1000,
    event_batch_size=100,
    num_workers=4,
    worker_prefetch=2,
    verbose=True,
)

graph = dataset[0]
print(graph.x.shape)
print(graph.edge_index.shape)
```

Each clustering merge creates two children. Parent–child edges are stored in
both directions, so an unpruned tree with `N` nodes has `2 × (N - 1)` directed
edges.

### Spatial graphs

```python
dataset = JetGraphProducer(
    root="data/",
    output_dataset_name="qcd_spatial",
    use_lund_decomp=False,
    delta_r_threshold=0.2,
    save_edge_attributes=True,
    graphs_per_chunk=1000,
)
```

Spatial graphs store constituent `(η, φ)` coordinates in `graph.pos`.

## Multiprocessing and storage tuning

The two sizing options are independent:

- `event_batch_size` controls how many ROOT entries are assigned to one worker
  task. Smaller batches improve load balancing but increase scheduling and file
  opening overhead.
- `graphs_per_chunk` controls how many completed graphs are stored in each
  `.pt` file. Larger chunks reduce file count but increase load-time memory.

`num_workers=1` uses the sequential path. A good starting point for a local
machine is:

```python
num_workers=4
event_batch_size=100
graphs_per_chunk=1000
worker_prefetch=2
```

Benchmark these values on the target filesystem: ROOT I/O can become the
bottleneck before CPU clustering does. Workers read their own ROOT ranges;
only the parent process writes graph chunks and metadata. Results are consumed
in deterministic event order.

The deprecated names `chunk_size` and `processing_batch_size` remain accepted
as aliases for `graphs_per_chunk` and `event_batch_size`.

## Processed dataset layout and cache validation

For a dataset named `qcd_lund`, processing creates:

```text
data/processed/qcd_lund_meta.pt
data/processed/qcd_lund_chunk_0.pt
data/processed/qcd_lund_chunk_1.pt
...
```

Constructing `JetGraphProducer` again with the same name loads these files
instead of reprocessing them. Graph-affecting options are stored in metadata
and validated on load. Reusing a name with incompatible settings raises an
error that lists the differing options.

Use a new `output_dataset_name` for a different representation or
configuration. Legacy caches without configuration metadata must be
regenerated.

## Loading datasets

`GraphDataLoader` separates two concepts:

1. A fixed permutation defines reproducible train/validation/test partitions.
2. A sampling mode controls ordering within one selected partition.

This makes it possible to mix detector periods before splitting while still
reshuffling the training partition independently each epoch:

```python
import torch

from data_loading.graphDataLoader import GraphDataLoader

paths = [
    ["data/", "qcd_lund"],
    ["data/", "signal_lund"],
]

train = GraphDataLoader(
    dataset_paths=paths,
    split_windows=[[0.0, 0.8], [0.0, 0.8]],
    split_seed=12345,
    sampling="weighted",
    batch_size=128,
)

split_state = train.get_split_state()
torch.save(split_state, "dataset_split.pt")

validation = GraphDataLoader(
    dataset_paths=paths,
    split_windows=[[0.8, 0.9], [0.8, 0.9]],
    split_state=split_state,
    sampling="sequential",
    batch_size=128,
)

test = GraphDataLoader(
    dataset_paths=paths,
    split_windows=[[0.9, 1.0], [0.9, 1.0]],
    split_state=split_state,
    sampling="sequential",
    batch_size=128,
)

for batch in train:
    # Move batches to the accelerator in the training loop.
    batch = batch.to("cuda")
    ...
```

To reproduce the split in another process:

```python
split_state = torch.load("dataset_split.pt", weights_only=False)
```

Split states contain path-qualified dataset keys, lengths, complete
permutations, and metadata fingerprints. A split is rejected if the processed
dataset has been regenerated or otherwise changed.

The fixed permutation determines which graphs belong to each partition.
Within a partition, the selected indices are traversed in storage order so
sequential statistics and evaluation passes retain chunk locality. This does
not change partition membership. `sampling="shuffle"` randomizes access within
the selected partition. Weighted sampling first makes the exact weighted draw
with replacement, then randomizes and groups that draw by storage chunk. This
preserves all sampled graphs and multiplicities while avoiding repeated chunk
deserialization.

Available sampling modes are:

- `sampling="sequential"`: stable partition order.
- `sampling="shuffle"`: reshuffle the selected partition every epoch.
- `sampling="weighted"`: sample with replacement using saved event weights.

Set `sampling_weights="uniform"` to use equal probabilities with weighted
sampling. `dataset_fraction` can assign a target sampling fraction to each
dataset; it is deliberately separate from the physics weights stored on each
graph.

Use `split_seed=None` when the original dataset order should be preserved.
If `jet_idx` filtering is needed, produce the dataset with
`save_jet_idx=True`.

The older `window_to_load`, `permutation`, `shuffle`, and `sample` arguments
remain available as deprecated compatibility aliases.

## Normalization

`OnTheFlyNormalizer` is a PyG transform and is applied lazily when a graph is
accessed:

```python
from utils.LundTreeUtilities import OnTheFlyNormalizer

normalizer = OnTheFlyNormalizer(["x"], means, stds)
loader = GraphDataLoader(
    [["data/", "qcd_lund"]],
    normalizer=normalizer,
    batch_size=128,
)
```

Means and standard deviations should be computed from the training partition
only.

## Autoencoder models

The vanilla GCN autoencoder and the small graph-transformer alternative expose
the same forward, loss, and per-graph scoring methods:

```python
from architectures.GraphAE import graphAE
from architectures.GraphTransformerAE import GraphTransformerAE

gcn_model = graphAE(
    num_features=n_features,
    architecture=(64, 32, 16),
)

transformer_model = GraphTransformerAE(
    num_features=n_features,
    architecture=(64, 32, 16),
    heads=2,
    dropout=0.1,
)

# Select either model without changing the existing training loop.
model = transformer_model.to(device)
```

The transformer uses `TransformerConv` attention over the graph edges rather
than dense all-pairs attention. Its feature decoder mirrors the encoder widths,
while edge reconstruction uses the same inner-product decoder as `graphAE`.
Because both models implement `edges_loss`, `features_loss`, `score_batch`, and
`score_event`, they can also be passed directly to `RootGraphScorer`.

## Scoring ROOT files

`RootGraphScorer` streams bounded entry ranges directly from ROOT through graph
construction and batched inference. It creates no intermediate `.pt` files and
returns one ordered, jagged score array per input file:

```python
from utils.GraphBuilder import GraphBuildConfig
from utils.RootGraphScorer import RootGraphScorer

graph_config = GraphBuildConfig(
    use_lund_decomp=True,
    n_lund_vars=5,
    save_4vectors_in_lund_tree=True,
    save_energy_fractions=True,
)

scorer = RootGraphScorer(
    model,
    graph_config,
    normalizer={"means": means, "stds": stds},
    alpha=1.0,
    device="cuda",
    root_entry_batch_size=1000,
    graph_batch_size=256,
    num_workers=4,
    worker_prefetch=2,
    verbose=True,
)

scores = scorer.score_files(["input_a.root", "input_b.root"])
first_event_scores = scores["input_a.root"][0]
```

Each result has shape `events × jets`, matching the original entry and jet
order. A jet for which no graph can be constructed receives `NaN`, preserving
alignment. For pipelines that should not retain all scores in memory, use
`scorer.iter_file(...)` to consume contiguous `ScoreChunk` objects.

`num_workers` parallelizes ROOT reading and graph construction over contiguous
entry ranges. Worker results are consumed in submission order, while model
inference remains in the parent process, so event and jet ordering is
unchanged. `worker_prefetch` bounds the number of queued ranges per worker.
Use `num_workers=1` for the sequential path. Multiprocessing uses `spawn` by
default so workers do not inherit initialized PyTorch or CUDA thread state.
The start method can be overridden with `multiprocessing_start_method`.
When using multiple workers from a standalone Python script, create and run
the scorer inside the usual `if __name__ == "__main__":` guard required by
the `spawn` start method. This guard is not needed in a notebook.

The inference score is the positive-edge reconstruction loss plus `alpha`
times the feature reconstruction loss. Negative edges remain part of training
but are deliberately excluded from inference for now; this choice can be
revisited if false-positive edge predictions become useful.

For debugging one already-built graph without creating a PyG batch:

```python
model.eval()
with torch.inference_mode():
    total, edge, features = model.score_event(qcd[0], alpha=1.0)
```

`score_event` also accepts `model.score_event(graph.x, graph.edge_index)`.

The older `ScoreProducer` workflow remains available for writing a complete
copy of the ROOT file with an added score branch.

## Tests

Run the complete regression suite from the repository root:

```bash
python -m unittest discover -s tests -v
```

The tests create temporary PFNanoAOD fixtures and cover:

- Lund feature configurations and binary-tree invariants.
- Pruning and single-node trees.
- Spatial φ wrapping.
- Sequential/parallel equivalence.
- Cache mismatch and legacy-cache rejection.
- Chunk sizing aliases.
- Loader filtering and sampler weights.
- Explicit rejection of removed TreeMaker input.

### Repository layout

```text
run_pipeline.py             Master orchestrator for end-to-end GAE pipeline
produce_graphs.py           Stage 1: Graph dataset production and caching
check_produced_shards.py    Verify integrity and completeness of produced graph shards
train_gae.py                Stage 2: Autoencoder model training and checkpointing
score_gae.py                Stage 3: Streaming inference on ROOT files
plot_gae.py                 Stage 4: Loss curves, ROC/SIC discrimination plots
gae_config.json             Central pipeline, model, and sample configuration
setup_gae_env.sh            Environment initialization wrapper for LCG view
architectures/              GNN model definitions (GraphAE, GraphTransformerAE)
condor/                     HTCondor batch submission scripts and templates
crab/                       CRAB grid job configs and wrappers
data_loading/               Lazy graph DataLoader with weighted/chunk sampling
inputJson/                  Sample file lists and xrootd endpoints
tests/                      Regression and unit test suite
utils/Cuts.py               Configurable event selections (Control / Signal regions)
utils/GraphBuilder.py       Shared PFNanoAOD graph construction
utils/JetGraphProducer.py   Dataset production, caching, multiprocessing
utils/LundTreeUtilities.py  Cambridge–Aachen clustering and tree utilities
utils/RootGraphScorer.py    Streaming, ordered ROOT inference
utils/ScoreProducer.py      ROOT scoring workflow
```

## GAE training pipeline

The Graph Autoencoder (GAE) pipeline reads PFNanoAOD ROOT files listed in
`inputJson/`, applies physics selections, trains a graph autoencoder on the
unsupervised data control region, scores signal, background, and data files, and
produces discrimination and evaluation plots.

The pipeline is organized into 4 modular steps coordinated by a master Python
orchestrator (`run_pipeline.py`):

1. **Graph production (`produce_graphs.py`)**: Builds and caches chunked PyG graph datasets.
2. **Model training (`train_gae.py`)**: Trains the autoencoder with preemption safety, checkpointing, and optional Comet ML tracking.
3. **Streaming scoring (`score_gae.py`)**: Evaluates the trained model on ROOT files without writing intermediate graph files.
4. **Discrimination plotting (`plot_gae.py`)**: Computes loss curves, score distributions, ROC curves, and Significance Improvement Characteristic (SIC) plots.

Everything is driven by `gae_config.json` and can be overridden via CLI flags.

---

### Environment & Comet ML setup

The base pipeline runs directly in CERN's LCG environment:

```bash
cd SVJ_GNN
source setup_gae_env.sh
```

To enable optional online experiment tracking with [Comet ML](https://www.comet.com/):

```bash
# 1. Create a virtual environment layered on top of the LCG view once:
python -m venv --system-site-packages .venv-gae
source .venv-gae/bin/activate
python -m pip install -r requirements-gae.txt

# 2. Export your Comet credentials:
export COMET_API_KEY="your-api-key"
export COMET_WORKSPACE="your-workspace"
export COMET_PROJECT_NAME="SVJScouting_GAE"
```

If `comet-ml` is not installed or `--no-comet` is passed, the pipeline runs
normally with local console and JSON history logging.

---

### Regions & physics selections

Selections are defined centrally in `utils/Cuts.py`. To ensure clean anomaly
detection and avoid signal contamination:

- **Training** is performed exclusively on **Data in the Control Region**:
  - `1.5 < |Δη(j0, j1)| < 2.2` between the two leading FatJets.
  - Leading two FatJets must satisfy `FatJet_isGood`.
  - 2018 post-HEM veto: reject leading FatJet with $p_T > 400\text{ GeV}$ and
    $\text{photonEnergyFraction} > 0.6$. Post-HEM MC events use the `HEMVeto` branch.
- **Scoring and Discrimination** can be evaluated in either:
  - `--region control` (default): $1.5 < |\Delta\eta| < 2.2$ (the control region).
  - `--region signal_region` (or `signal`): $|\Delta\eta| < 1.5$ (the signal region).
  - Additional supported regions: `validation`, `highRT`, `lowRT`, and `inclusive`.

A fingerprint of the selection settings is embedded in dataset metadata. If
cuts change, cached graph datasets are invalidated automatically.

---

### Master orchestrator (`run_pipeline.py`)

`run_pipeline.py` provides a unified CLI to execute the entire analysis or
individual stages in sequence:

```bash
# Run the complete end-to-end pipeline:
python run_pipeline.py --all --epochs 20 --device auto

# Run only production and training:
python run_pipeline.py --steps produce train --epochs 10

# Score and generate discrimination plots in the Signal Region:
python run_pipeline.py --steps score plot --region signal_region \
    --checkpoint training/GraphAE_graphAE/checkpoint_best.pt

# Quick smoke test on CPU with minimal events:
python run_pipeline.py --all --train-on smoke --epochs 2 --device cpu --no-comet
```

#### Orchestrator CLI options

| Flag | Description | Default |
|---|---|---|
| `--all` | Execute all 4 stages: `produce`, `train`, `score`, `plot` | `False` |
| `--steps` | Selective stages to run (e.g. `--steps train score`) | None |
| `--config` | Path to JSON config file | `gae_config.json` |
| `--train-on` | Training sample(s) | `data` control region |
| `--region` | Selection region for scoring/plotting (`control`, `signal_region`, `inclusive`, etc.) | `control` |
| `--device` | Execution device (`auto`, `cuda`, `cpu`) | `auto` |
| `--epochs` | Total training epochs | Config setting |
| `--resume` | Resume training from checkpoint (`[auto\|PATH]`) | None |
| `--add-epochs` | Additional epochs to train beyond checkpoint count | None |
| `--batch-size` | Batch size override | Config setting |
| `--accumulate-grad-batches` | Gradient accumulation steps for limited GPU memory | `1` |
| `--model-spec` | Model architecture override (e.g. `GraphTransformerAE.GraphTransformerAE`) | Config setting |
| `--model-kwargs` | JSON string overriding model constructor arguments | None |
| `--comet` / `--no-comet` | Enable or disable Comet ML logging | Auto-detected |
| `--num-workers` | Worker processes for ROOT reading and graph construction | Config setting |

---

### Checkpoint resumption & preemption safety

`train_gae.py` and `run_pipeline.py` support robust state persistence, allowing
training to pause and resume seamlessly across batch job preemption or extended
training schedules:

#### Checkpoint contents
Checkpoints preserve the full training state:
- `state_dict`: Model weights and biases.
- `optimizer_state_dict`: Optimizer momentum buffers, moved to the active device.
- `epoch`: Completed epoch count.
- `history`: Full loss trajectory (`batch_total`, `batch_edges`, `batch_nodes`, `epoch_train`, `epoch_validation`, `test`).
- `best_val_loss`: Lowest observed validation loss.
- `means` & `stds`: Cached feature normalization statistics (skips recalculation on resume).
- `split_state`: Reproducible dataset train/val/test partitions.

#### Checkpoint files saved
- `checkpoint_latest.pt`: Saved after every epoch. Preempted HTCondor jobs can restart immediately from the last completed epoch.
- `checkpoint_best.pt`: Saved whenever validation loss improves.
- `checkpoint.pt`: Canonical final checkpoint saved upon completion.

#### How to resume
```bash
# Auto-detect latest checkpoint and continue to 20 total epochs:
python run_pipeline.py --steps train --resume --epochs 20 --run-name my_run

# Add 5 more epochs beyond what was previously trained:
python run_pipeline.py --steps train --resume --add-epochs 5 --run-name my_run

# Resume directly using train_gae.py:
python train_gae.py --config gae_config.json --resume training/my_run/checkpoint_latest.pt --add-epochs 10
```

---

### Limited GPU memory & CPU training

For machines or batch slots with limited GPU memory:
- Use gradient accumulation:
  ```bash
  python run_pipeline.py --steps train --batch-size 64 --accumulate-grad-batches 4 --device cuda
  ```
- Or train directly on CPU:
  ```bash
  python run_pipeline.py --steps train --device cpu --no-comet
  ```

---

### Running individual pipeline steps

Each pipeline step can also be run independently as a standalone script:

#### Step 1: Graph dataset production (`produce_graphs.py`)
```bash
# Produce graph cache for the training data sample:
python produce_graphs.py --samples data_2018 --num-workers 4

# Run with event limit for testing:
python produce_graphs.py --samples smoke --max-events 500 --num-workers 1
```

#### Step 2: Autoencoder training (`train_gae.py`)
```bash
# Train on data control region for 20 epochs:
python train_gae.py --train-on data_2018 --epochs 20 --device cuda

# Resume from latest checkpoint with additional epochs:
python train_gae.py --train-on data_2018 --resume --add-epochs 10
```

#### Step 3: Streaming ROOT scoring (`score_gae.py`)
```bash
# Score data, background, and signal samples in the signal region:
python score_gae.py \
    --checkpoint training/GraphAE_graphAE/checkpoint_best.pt \
    --samples data_2018 qcd_2018 signal_2018 \
    --region signal_region \
    --output-dir output_scores
```
`score_gae.py` streams directly from ROOT entries without writing intermediate
`.pt` graph files, outputting compact ROOT files with score branches under `output_scores/`.

#### Step 4: Discrimination & evaluation plotting (`plot_gae.py`)
```bash
# Generate ROC curves, SIC plots, and loss distributions:
python plot_gae.py \
    --scores-dir output_scores \
    --history training/GraphAE_graphAE/history.json \
    --plots-dir plots

# Include per-feature reconstruction residuals and latent-space distributions:
python plot_gae.py \
    --scores-dir output_scores \
    --checkpoint training/GraphAE_graphAE/checkpoint_best.pt \
    --residual-sample data_2018 \
    --plots-dir plots
```
Generates loss curves, reconstruction loss histograms, ROC curves, and SIC
curves (comparing signal points against background), along with a `histograms.root`
containing all `TH1` objects styled with `mplhep`.

---

### Model registry & architecture configuration

Architectures are dynamically resolved by `architectures/ModelRegistry.py`.
To test different models, modify `gae_config.json` or pass CLI flags:

```bash
# Use Graph Transformer autoencoder:
python run_pipeline.py --steps train \
    --model-spec GraphTransformerAE.GraphTransformerAE \
    --model-kwargs '{"heads": 4, "dropout": 0.1}'
```

Available models in `architectures/`:
- `GraphAE.graphAE`: GCN encoder with inner-product adjacency decoder and MLP feature decoder.
- `GraphTransformerAE.GraphTransformerAE`: Multi-head `TransformerConv` attention encoder.

---

### Batch submission (HTCondor & CRAB)

#### HTCondor
`condor/submit_condor.py` generates submit descriptions and scripts for distributed execution:

```bash
# Submit graph production across worker nodes (sharded by files):
python condor/submit_condor.py --stage produce --samples background_2018 --files-per-job 5
condor_submit condor/submit/produce.jdl

# After produce jobs finish, verify all shards and chunk files:
python condor/submit_condor.py --stage produce --verify --samples background_2018 --files-per-job 5

# Submit training on a GPU slot:
python condor/submit_condor.py --stage train --request-gpus 1
condor_submit condor/submit/train.jdl

# Submit scoring:
python condor/submit_condor.py --stage score --samples signal_2018 --files-per-job 5
condor_submit condor/submit/score.jdl
```

#### Verifying produced shards and resubmitting failed jobs

After Condor graph production jobs finish, verify that all expected shards are present, complete, and uncorrupted:

```bash
# Verify using condor/submit_condor.py:
python condor/submit_condor.py --stage produce --verify \
    --samples background_2018 \
    --files-per-job 5 \
    --config gae_config_aimar.json

# Or using the standalone check_produced_shards.py tool:
python check_produced_shards.py --config gae_config_aimar.json \
    --samples background_2018 \
    --files-per-job 5

# Or recursively scan any EOS directory / subdirectories directly:
python check_produced_shards.py /eos/user/n/nparmar/SVJScouting/data_aimar_data_v2
```

The verification tool checks:
- **Shard completeness**: Compares actual shards against expected totals calculated from the input JSON. Detects missing shards and incomplete shards (e.g. jobs that were evicted before writing `_meta.pt`).
- **File integrity**: Concurrently inspects ZIP archive central directories and file sizes over 32 parallel threads (~60 s for ~4,000 files on EOS), catching truncated files and `PytorchStreamReader` central directory errors.
- **Automatic resubmission**: If any failed or corrupted shards are detected, it automatically writes `condor/submit/produce_resubmit.jdl` queued with **only** the failed shards:
  ```bash
  condor_submit condor/submit/produce_resubmit.jdl
  ```

#### CRAB
Per-file production and scoring are supported through `crab/crabConfig_gae.py`,
which sources input files from `inputJson/` and executes `crab/crab_script.sh`.

---

## Contact

roberto.seidita@cern.ch

