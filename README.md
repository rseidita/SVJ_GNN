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

## Repository layout

```text
architectures/              GNN model definitions
data_loading/               Lazy graph DataLoader
tests/                      Regression suite
utils/GraphBuilder.py       Shared PFNanoAOD graph construction
utils/JetGraphProducer.py   Dataset production, caching, multiprocessing
utils/LundTreeUtilities.py  Cambridge–Aachen clustering and tree utilities
utils/RootGraphScorer.py    Streaming, ordered ROOT inference
utils/ScoreProducer.py      ROOT scoring workflow
```

## GAE training pipeline

A set of command line scripts wraps the modules above into a pipeline that
reads ROOT files listed in `inputJson/`, applies one shared selection to
signal, background and data, trains a graph autoencoder, and makes the
performance plots. It runs locally, under HTCondor, or under CRAB.

Everything is driven by `gae_config.json`: the graph representation, the
production and training settings, the model, and the samples. Each sample
declares a role (`signal`, `background` or `data`), a year, an input JSON,
and whether to use cross section weights. An optional `datasets` list
restricts a sample to some of the datasets inside its JSON.

### Input files

`inputJson/<type>/<year>_<type>.json` lists the skims on the GridKA dCache,
grouped by dataset (`Run2018A`, `QCD_HT500to700`,
`s-channel_mMed-1000_mDark-20_rinv-0.3`, ...). Regenerate them with a valid
grid proxy:

```bash
voms-proxy-init -voms cms
python make_input_json.py                     # 2017 and 2018, all types
python make_input_json.py --years 2018 --types background
```

Only 2017 and 2018 are listed; the 2016 background skims lack the scouting
collections. Cross section weights use the generated events of the dataset a
file belongs to, summed over all files of that dataset, so QCD HT bins or
signal points can share one sample.

### Environment

The LCG view provides every dependency except `comet-ml`:

```bash
cd SVJ_GNN
source setup_gae_env.sh
```

To add comet logging, create a virtual environment on top of the view once.
`setup_gae_env.sh` then picks it up automatically:

```bash
python -m venv --system-site-packages .venv-gae
source .venv-gae/bin/activate
python -m pip install -r requirements-gae.txt
```

Without it the pipeline still runs; `train_gae.py` falls back to local-only
logging, and `--no-comet` selects that explicitly.

### Selections

`utils/Cuts.py` is the only place selections are defined; both production
and scoring call `event_cut_mask(events, role, year)`. On top of the
pre-selection skims it requires:

- `FatJet_isGood` on the two leading FatJets;
- `1.5 < |Δη(j0, j1)| < 2.2` between the leading and subleading FatJet: the
  control region the model is trained on. It is applied to data, background
  and signal alike, so inference happens in the same region;
- 2018 post-HEM events: reject a leading FatJet with `pT > 400 GeV` and
  `photonEnergyFraction > 0.6` (extended gap-jet veto). Post-HEM is
  `run >= 319077` for data; for MC a deterministic pseudo-random fraction
  `POST_HEM_FRACTION` of events, keyed on `(run, lumSec, event)`;
- 2018 MC only: the `HEMVeto` branch is required for that post-HEM fraction.
  Data skims already include the veto.

Branches a cut needs are returned by `cut_branches(role, year)` so they are
read from the file. Changing a selection changes the fingerprint stored in
the processed dataset metadata, so stale graph caches are rejected instead of
being silently reused.

Both `produce_graphs.py` and `score_gae.py` print a cutflow per dataset
(cumulative events, absolute and relative efficiency) for the files they
process; `--no-cutflow` skips the extra pass in production.

Samples with `"per_dataset": true` (the signal entries) are expanded into one
sample per dataset, `signal_2018_s-channel_mMed-1000_mDark-20_rinv-0.3` and so
on, so each signal point is processed and plotted separately. The base name
still selects all of them: `--samples signal_2018`. Training uses **data**
in the control region only. QCD and signal are **not** used in
`train_gae.py`; they are scored after training and compared in
`plot_gae.py` (loss distributions, ROC, SIC).

### Running the pipeline

```bash
# 1. Build graph datasets for the training sample (data control region).
python produce_graphs.py --samples data_2018

# 2. Train the autoencoder on it.
python train_gae.py --train-on data_2018 --epochs 20

# 3. Score data, background and every signal point straight from ROOT.
python score_gae.py --checkpoint training/GraphAE_graphAE/checkpoint.pt \
    --output-dir output_scores \
    --samples data_2018 qcd_2018 signal_2018

# 4. Make the plots (combines data_part-*, qcd_part-*, per signal point).
python plot_gae.py --scores-dir output_scores \
    --history training/GraphAE_graphAE/history.json
```

`score_gae.py` writes one local file per input ROOT file under
``output_scores/`` (``data_part-0.root``, ``qcd_part-0.root``,
``signal_<point>_part-0.root``). A small end-to-end test with two data
files is ``bash run_test_pipeline.sh`` (use ``SKIP_PRODUCE=1 SKIP_TRAIN=1``
to rerun scoring and plots only).

Step 4 writes the loss curves, the reconstruction-loss distributions for
signal, background and data, the normalized number of surviving events
against the loss threshold, the ROC curves and the significance improvement
against background efficiency (one curve per signal point against the pooled
background), together with a `histograms.root` holding the TH1 versions. Adding `--checkpoint` and `--residual-sample` also produces the
per-feature reconstruction residuals and the latent-space distributions.

A quick end-to-end check on `part-14.root` in the parent directory (2018
scouting **data**, not MC, listed in `inputJson/smoke/2018_data.json`). Data
files have no `Metadata` tree, so the `smoke` sample uses `weights: null`;
only MC samples should set `weights: "xsec"`.

```bash
python produce_graphs.py --samples smoke --max-events 200 --num-workers 1
python train_gae.py --train-on smoke --epochs 1 --no-comet
python score_gae.py --checkpoint training/GraphAE_graphAE/checkpoint.pt --samples smoke
python plot_gae.py --history training/GraphAE_graphAE/history.json
```

`--train-on smoke` is only a plumbing check with one file. For a real run,
train on the data control region (`data_2017`, `data_2018`). Score
`qcd_2018` or `background_2018` and `signal_2018` afterward for
discrimination plots; ROC and SIC need both a scored signal and a scored
background sample.

### Choosing a model

The pipeline never names an architecture. `architectures/ModelRegistry.py`
resolves a `"Module.ClassName"` spec inside `architectures/`, so a new model
only needs a new file there and a one line change in `gae_config.json`:

```json
"model": {"spec": "MyNewAE.myNewAE", "kwargs": {"latent_dim": 8}}
```

`kwargs` is passed straight to the constructor, so a model with different
arguments needs no pipeline change. A model must provide `forward`,
`score_batch`, and either `training_loss(batch, alpha)` or both `edges_loss`
and `features_loss`. Defining `training_loss` lets a model use its own
objective; everything else falls back to `edges_loss + alpha * features_loss`.

### Batch submission

Graph production and scoring are one job per shard of a sample's files.
Training is a single job, on a GPU when one is requested.

```bash
python condor/submit_condor.py --stage produce --samples background_2018 \
    --files-per-job 5
python condor/submit_condor.py --stage train --request-gpus 1
condor_submit condor/submit/produce.jdl
```

CRAB covers the same per-file production and scoring stages through
`crab/crabConfig_gae.py`, which feeds `Data.userInputFiles` from the sample's
input JSON and runs `crab/crab_script.sh` as its `scriptExe`. Each job passes
its files with `--files`, leaving the input JSON intact so cross section
weights still see the whole dataset. Training is deliberately not a CRAB
workflow: a long GPU job does not fit the model.

## Contact

roberto.seidita@cern.ch
