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


ROLES = ("signal", "background", "data")


@dataclasses.dataclass
class SampleConfig:
    """One input sample: a role, a file list, and how to weight it."""

    name: str
    role: str
    file_list: str
    weights: str = None
    label: float = 0.0
    dataset_name: str = None

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
        if self.dataset_name is None:
            self.dataset_name = self.name


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

    samples = {}
    for name, sample in raw.get("samples", {}).items():
        samples[name] = SampleConfig(name=name, **sample)

    return PipelineConfig(
        graph=raw.get("graph", {}),
        production=raw.get("production", {}),
        training=raw.get("training", {}),
        model=ModelConfig(**raw.get("model", {})),
        samples=samples,
        path=path,
    )
