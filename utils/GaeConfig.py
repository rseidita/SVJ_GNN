"""Configuration shared by the GAE production, training, scoring and plotting
command line scripts.

The configuration is a single JSON file, ``gae_config.json`` by default. It
declares the graph representation, the production and training settings, the
model to use, and the samples split into the three roles ``signal``,
``background`` and ``data``.
"""

import dataclasses
import glob
import json
import os
import re

from utils.GraphBuilder import GraphBuildConfig
from utils.InputJson import dataset_groups, flat_files, load_input_json


ROLES = ("signal", "background", "data")


@dataclasses.dataclass
class SampleConfig:
    """One input sample: a role, a year, an input JSON, and its weighting.

    ``input_json`` is resolved relative to the configuration file. The
    optional ``datasets`` list restricts the sample to some of the datasets
    inside the JSON (for example a single QCD HT bin or one signal point).

    With ``per_dataset`` true, ``load_config`` expands the entry into one
    sample per dataset in the JSON, named ``<name>_<dataset>``, so every
    signal point is produced, scored and plotted on its own. The original
    name is kept as ``group`` and selects all of them on the command line.
    """

    name: str
    role: str
    year: str
    input_json: str
    weights: str = None
    label: float = 0.0
    dataset_name: str = None
    datasets: list = None
    per_dataset: bool = False
    group: str = None
    config_dir: str = "."

    def __post_init__(self):
        if self.role not in ROLES:
            raise ValueError(
                f"Sample {self.name!r} has role {self.role!r}, "
                f"expected one of {ROLES}"
            )
        if self.weights not in ("xsec", None):
            raise ValueError(
                f"Sample {self.name!r} has weights {self.weights!r}, "
                "expected 'xsec' or null"
            )
        self.year = str(self.year)
        if self.dataset_name is None:
            self.dataset_name = self.name
        if self.group is None:
            self.group = self.name

    def expand_per_dataset(self):
        """One sample per dataset of the JSON, or ``[self]`` if not requested."""
        if not self.per_dataset:
            return [self]
        return [
            dataclasses.replace(
                self,
                name=f"{self.name}_{dataset}",
                dataset_name=None,
                datasets=[dataset],
                per_dataset=False,
                group=self.name,
            )
            for dataset in self.groups()
        ]

    @property
    def input_json_path(self):
        if os.path.isabs(self.input_json):
            return self.input_json
        return os.path.normpath(os.path.join(self.config_dir, self.input_json))

    def groups(self):
        """``{dataset: [files]}`` of this sample."""
        return dataset_groups(
            load_input_json(self.input_json_path), self.datasets
        )

    def files(self):
        return flat_files(self.groups())


@dataclasses.dataclass
class ModelConfig:
    """Which architecture to build, and with which constructor arguments."""

    spec: str = "GraphAE.graphAE"
    kwargs: dict = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class PipelineConfig:
    graph: dict
    production: dict
    training: dict
    model: ModelConfig
    samples: dict
    scoring: dict = dataclasses.field(default_factory=dict)
    plotting: dict = dataclasses.field(default_factory=dict)
    path: str = None

    def sample(self, name):
        if name not in self.samples:
            raise KeyError(
                f"Unknown sample {name!r}; configured samples are "
                f"{sorted(self.samples)}"
            )
        return self.samples[name]

    def select(self, names=None):
        """Samples matching *names*: exact names or ``per_dataset`` groups.

        With no names, every sample.
        """
        if not names:
            return [self.samples[name] for name in sorted(self.samples)]
        selected = []
        for name in names:
            if name in self.samples:
                selected.append(self.samples[name])
                continue
            group = [
                sample
                for sample in self.samples.values()
                if sample.group == name
            ]
            if not group:
                raise KeyError(
                    f"Unknown sample or group {name!r}; configured samples "
                    f"are {sorted(self.samples)}"
                )
            selected.extend(sorted(group, key=lambda sample: sample.name))
        return selected

    def samples_with_role(self, role):
        if role not in ROLES:
            raise ValueError(f"Unknown role {role!r}, expected one of {ROLES}")
        return [
            sample
            for sample in self.samples.values()
            if sample.role == role
        ]

    def graph_build_config(self, label=0.0):
        """The GraphBuildConfig implied by the shared graph settings."""
        allowed = {field.name for field in dataclasses.fields(GraphBuildConfig)}
        kwargs = {
            key: value
            for key, value in self.graph.items()
            if key in allowed
        }
        kwargs["label"] = label
        return GraphBuildConfig(**kwargs)

    def dataset_root(self, sample, base_dir=None, require_exists=False):
        """Output directory holding the processed graphs of *sample*."""
        prod_dir = self.production.get("output_dir", "data")
        candidates = []
        base = base_dir or os.environ.get("GAE_DATA_DIR")
        if base:
            norm_base = os.path.normpath(base)
            base_leaf = os.path.basename(norm_base)
            if base_leaf in ("data", "data_aimar") and base_leaf != prod_dir:
                raise ValueError(
                    f"Data folder mismatch: base_dir '{base}' ends with '{base_leaf}', "
                    f"but pipeline config specifies production directory '{prod_dir}'. "
                    f"Please specify the matching data folder or adjust the config."
                )
            if base_leaf == prod_dir:
                candidates.append(os.path.join(base, sample.name))
            else:
                candidates.append(os.path.join(base, prod_dir, sample.name))

        candidates.append(os.path.join(prod_dir, sample.name))

        eos_base = os.environ.get("GAE_OUTPUT_EOS") or "/eos/user/n/nparmar/SVJScouting"
        if eos_base and os.path.isdir(eos_base):
            norm_eos = os.path.normpath(eos_base)
            cand = os.path.join(eos_base, sample.name) if os.path.basename(norm_eos) == prod_dir else os.path.join(eos_base, prod_dir, sample.name)
            if cand not in candidates:
                candidates.append(cand)

        for cand in candidates:
            if os.path.isdir(os.path.join(cand, "processed")):
                return cand

        if require_exists:
            checked = ", ".join(f"'{c}'" for c in candidates)
            raise FileNotFoundError(
                f"Dataset for sample '{sample.name}' was not found in configured production directory '{prod_dir}'. "
                f"Checked locations: {checked}. "
                f"Automatic fallback across different data folders is disabled to prevent using incorrect datasets."
            )

        return candidates[0]

    def all_dataset_paths(self, sample, base_dir=None, require_exists=True):
        """List of ``[root, dataset_name]`` pairs for *sample*, expanding all shards if present."""
        root = self.dataset_root(sample, base_dir=base_dir, require_exists=require_exists)
        processed_dir = os.path.join(root, "processed")
        base_name = sample.dataset_name

        shard_pattern = os.path.join(processed_dir, f"{base_name}_shard*_meta.pt")
        shard_meta_files = glob.glob(shard_pattern)
        if shard_meta_files:
            def shard_key(p):
                m = re.search(r"_shard(\d+)_meta\.pt$", p)
                return int(m.group(1)) if m else 0
            shard_meta_files.sort(key=shard_key)
            return [[root, os.path.basename(smf)[:-8]] for smf in shard_meta_files]

        if require_exists:
            meta_file = os.path.join(processed_dir, f"{base_name}_meta.pt")
            chunk_files = glob.glob(os.path.join(processed_dir, f"{base_name}_chunk_*.pt"))
            if not os.path.exists(meta_file) and not chunk_files:
                prod_dir = self.production.get("output_dir", "data")
                raise FileNotFoundError(
                    f"No processed graph files found for sample '{sample.name}' in '{processed_dir}'. "
                    f"Expected '{base_name}_meta.pt' or '{base_name}_chunk_*.pt' in production directory '{prod_dir}'."
                )

        return [[root, base_name]]

    def dataset_paths(self, sample, base_dir=None):
        """``[root, dataset_name]`` pair as expected by GraphDataLoader."""
        paths = self.all_dataset_paths(sample, base_dir=base_dir)
        return paths[0]

    def find_checkpoint(self, checkpoint_path=None, run_name=None, seed=None, base_dir=None):
        """Locate model checkpoint file from explicit path or previous train stage output."""
        if checkpoint_path:
            if os.path.isfile(checkpoint_path):
                return os.path.abspath(checkpoint_path)
            if os.path.isdir(checkpoint_path):
                for name in ("checkpoint_best.pt", "checkpoint.pt", "checkpoint_latest.pt"):
                    cand = os.path.join(checkpoint_path, name)
                    if os.path.isfile(cand):
                        return os.path.abspath(cand)
            raise FileNotFoundError(f"Specified checkpoint '{checkpoint_path}' does not exist.")

        train_dir = self.training.get("output_dir", "training")
        bases = []
        if base_dir:
            bases.append(base_dir)
        env_eos = os.environ.get("GAE_OUTPUT_EOS")
        if env_eos and env_eos not in bases:
            bases.append(env_eos)
        default_eos = "/eos/user/n/nparmar/SVJScouting"
        if os.path.isdir(default_eos) and default_eos not in bases:
            bases.append(default_eos)
        if "." not in bases:
            bases.append(".")

        candidate_bases = []
        for base in bases:
            if train_dir and os.path.isabs(train_dir):
                candidate_bases.append(train_dir)
            else:
                candidate_bases.append(os.path.join(base, train_dir))
            candidate_bases.append(base)

        spec_name = self.model.spec.replace(".", "_")
        run_candidates = []
        if run_name:
            if seed is not None and not run_name.endswith(f"_seed{seed}"):
                run_candidates.append(f"{run_name}_seed{seed}")
            run_candidates.append(run_name)
        if seed is not None:
            run_candidates.append(f"{spec_name}_seed{seed}")
        run_candidates.append(spec_name)

        for c_base in candidate_bases:
            if not os.path.isdir(c_base):
                continue
            for r in run_candidates:
                r_dir = os.path.join(c_base, r)
                if os.path.isdir(r_dir):
                    for ckpt_name in ("checkpoint_best.pt", "checkpoint.pt", "checkpoint_latest.pt"):
                        f = os.path.join(r_dir, ckpt_name)
                        if os.path.isfile(f):
                            return os.path.abspath(f)

        for c_base in candidate_bases:
            if not os.path.isdir(c_base):
                continue
            try:
                subdirs = [os.path.join(c_base, d) for d in os.listdir(c_base) if os.path.isdir(os.path.join(c_base, d))]
                subdirs.sort(key=lambda d: os.path.getmtime(d), reverse=True)
                for sdir in subdirs:
                    for ckpt_name in ("checkpoint_best.pt", "checkpoint.pt", "checkpoint_latest.pt"):
                        f = os.path.join(sdir, ckpt_name)
                        if os.path.isfile(f):
                            return os.path.abspath(f)
            except OSError:
                continue

        for c_base in candidate_bases:
            if not os.path.isdir(c_base):
                continue
            for ckpt_name in ("checkpoint_best.pt", "checkpoint.pt", "checkpoint_latest.pt"):
                f = os.path.join(c_base, ckpt_name)
                if os.path.isfile(f):
                    return os.path.abspath(f)

        raise FileNotFoundError(
            f"No checkpoint found in train output locations (searched: {candidate_bases}). "
            "Please ensure training has completed or supply --checkpoint explicitly."
        )


def load_config(path="gae_config.json"):
    with open(path) as config_file:
        raw = json.load(config_file)

    config_dir = os.path.dirname(os.path.abspath(path))
    samples = {}
    for name, sample in raw.get("samples", {}).items():
        base = SampleConfig(name=name, config_dir=config_dir, **sample)
        for expanded in base.expand_per_dataset():
            samples[expanded.name] = expanded

    return PipelineConfig(
        graph=raw.get("graph", {}),
        production=raw.get("production", {}),
        training=raw.get("training", {}),
        model=ModelConfig(**raw.get("model", {})),
        samples=samples,
        scoring=raw.get("scoring", {}),
        plotting=raw.get("plotting", {}),
        path=path,
    )
