"""Configuration shared by the GAE production, training, scoring and plotting
command line scripts.

The configuration is a single JSON file, ``gae_config.json`` by default. It
declares the graph representation, the production and training settings, the
model to use, and the samples split into the three roles ``signal``,
``background`` and ``data``.
"""

import dataclasses
import json
import os

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

    def dataset_root(self, sample):
        """Output directory holding the processed graphs of *sample*."""
        return os.path.join(
            self.production.get("output_dir", "data"), sample.name
        )

    def dataset_paths(self, sample):
        """``[root, dataset_name]`` pair as expected by GraphDataLoader."""
        return [self.dataset_root(sample), sample.dataset_name]


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
        path=path,
    )
