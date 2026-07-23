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

## Scoring ROOT files

`ScoreProducer` builds the same Lund representation through the shared graph
builder, evaluates a compatible graph-autoencoder model, and writes a
`FatJet_GNNScore` branch to a new PFNanoAOD ROOT file.

See `score_per_file_example.py` for the current command-line workflow. Model
feature dimensions and normalization statistics must match the graph
configuration used during training.

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
utils/ScoreProducer.py      ROOT scoring workflow
```

## Contact

roberto.seidita@cern.ch
