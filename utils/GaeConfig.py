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

    def dataset_root(self, sample, base_dir=None):
        """Output directory holding the processed graphs of *sample*."""
        prod_dir = self.production.get("output_dir", "data")
        base = base_dir or os.environ.get("GAE_DATA_DIR")
        if base:
            for cand in (
                os.path.join(base, prod_dir, sample.name),
                os.path.join(base, "data", sample.name),
                os.path.join(base, "data_aimar", sample.name),
                os.path.join(base, sample.name),
            ):
                if os.path.isdir(os.path.join(cand, "processed")):
                    return cand
            return os.path.join(base, prod_dir, sample.name)

        local_cand = os.path.join(prod_dir, sample.name)
        if os.path.isdir(os.path.join(local_cand, "processed")):
            return local_cand

        eos_base = os.environ.get("GAE_DATA_DIR") or os.environ.get("GAE_OUTPUT_EOS")
        if eos_base and os.path.isdir(eos_base):
            for sub in (prod_dir, "data", "data_aimar"):
                cand = os.path.join(eos_base, sub, sample.name)
                if os.path.isdir(os.path.join(cand, "processed")):
                    return cand
            cand = os.path.join(eos_base, sample.name)
            if os.path.isdir(os.path.join(cand, "processed")):
                return cand

        for sub in ("data", "data_aimar"):
            cand = os.path.join(sub, sample.name)
            if os.path.isdir(os.path.join(cand, "processed")):
                return cand

        return local_cand

    def all_dataset_paths(self, sample, base_dir=None):
        """List of ``[root, dataset_name]`` pairs for *sample*, expanding all shards if present."""
        root = self.dataset_root(sample, base_dir=base_dir)
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

        return [[root, base_name]]

    def dataset_paths(self, sample, base_dir=None):
        """``[root, dataset_name]`` pair as expected by GraphDataLoader."""
        paths = self.all_dataset_paths(sample, base_dir=base_dir)
        return paths[0]


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
