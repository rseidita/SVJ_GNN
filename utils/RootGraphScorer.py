import collections
import concurrent.futures
import multiprocessing
import os
import pickle
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import awkward as ak
import numpy as np
import torch
import uproot
from torch_geometric.data import Batch
from tqdm import tqdm

from utils.GraphBuilder import (
    GraphBuildConfig,
    build_pfnano_jet_graph,
    pfnano_branches,
)
from utils.InputJson import normalize_input_path


_SCORER_WORKER_CONFIG = None
_SCORER_WORKER_MAX_JETS = None
_SCORER_WORKER_FILE = None
_SCORER_WORKER_BRANCHES = None
_SCORER_WORKER_TREE = None


def _initialize_scorer_worker(
    graph_config, max_jets, file_name, branches
):
    global _SCORER_WORKER_CONFIG, _SCORER_WORKER_MAX_JETS
    global _SCORER_WORKER_FILE, _SCORER_WORKER_BRANCHES
    global _SCORER_WORKER_TREE
    _SCORER_WORKER_CONFIG = graph_config
    _SCORER_WORKER_MAX_JETS = max_jets
    _SCORER_WORKER_FILE = file_name
    _SCORER_WORKER_BRANCHES = branches
    _SCORER_WORKER_TREE = uproot.open(f"{file_name}:Events")
    torch.set_num_threads(1)


def _build_graphs(events, graph_config, max_jets):
    collection = graph_config.jet_collection
    jet_counts = []
    graphs = []
    positions = []
    for event_index, event in enumerate(events):
        n_jets = int(event[f"n{collection}"])
        if max_jets is not None:
            n_jets = min(n_jets, max_jets)
        jet_counts.append(n_jets)
        for jet_index in range(n_jets):
            graph, _ = build_pfnano_jet_graph(
                event, jet_index, graph_config
            )
            if graph is not None:
                graphs.append(graph)
                positions.append((event_index, jet_index))
    return jet_counts, graphs, positions


def _process_scorer_entry_range(task):
    if _SCORER_WORKER_CONFIG is None or _SCORER_WORKER_TREE is None:
        raise RuntimeError("Scoring worker was not initialized")
    entry_start, entry_stop = task
    try:
        events = _SCORER_WORKER_TREE.arrays(
            _SCORER_WORKER_BRANCHES,
            entry_start=entry_start,
            entry_stop=entry_stop,
            library="ak",
        )
        result = _build_graphs(
            events, _SCORER_WORKER_CONFIG, _SCORER_WORKER_MAX_JETS
        )
        # Explicit bytes avoid relying on PyTorch shared-memory reducers and
        # keep worker-to-parent transfer behavior consistent across systems.
        return pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to build graphs for entries "
            f"[{entry_start}, {entry_stop}) from {_SCORER_WORKER_FILE!r}"
        ) from exc


@dataclass(frozen=True)
class ScoreChunk:
    """Scores for one contiguous range of entries in one ROOT file."""

    file_name: str
    entry_start: int
    entry_stop: int
    scores: ak.Array


class RootGraphScorer:
    """Stream PFNanoAOD entries through graph construction and inference.

    ROOT entries and graphs are held in bounded in-memory batches. No
    intermediate PyTorch dataset or ``.pt`` files are produced.
    """

    def __init__(
        self,
        model,
        graph_config: GraphBuildConfig,
        normalizer=None,
        *,
        alpha: float = 1.0,
        device=None,
        root_entry_batch_size: int = 1000,
        graph_batch_size: int = 256,
        max_events: int | None = None,
        max_jets: int | None = None,
        num_workers: int = 1,
        worker_prefetch: int = 2,
        multiprocessing_start_method: str | None = None,
        verbose: bool = False,
    ):
        if root_entry_batch_size <= 0:
            raise ValueError("root_entry_batch_size must be positive")
        if graph_batch_size <= 0:
            raise ValueError("graph_batch_size must be positive")
        if max_events is not None and max_events < 0:
            raise ValueError("max_events must be nonnegative")
        if max_jets is not None and max_jets < 0:
            raise ValueError("max_jets must be nonnegative")
        if num_workers <= 0:
            raise ValueError("num_workers must be positive")
        if worker_prefetch <= 0:
            raise ValueError("worker_prefetch must be positive")
        if (
            multiprocessing_start_method is not None
            and multiprocessing_start_method
            not in multiprocessing.get_all_start_methods()
        ):
            raise ValueError(
                f"Unsupported multiprocessing start method "
                f"{multiprocessing_start_method!r}"
            )
        if not hasattr(model, "score_batch"):
            raise TypeError(
                "model must provide score_batch(x, edge_index, batch, alpha)"
            )

        self.device = torch.device(
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = model.to(self.device)
        self.model.eval()
        self.graph_config = graph_config
        self.alpha = float(alpha)
        self.root_entry_batch_size = int(root_entry_batch_size)
        self.graph_batch_size = int(graph_batch_size)
        self.max_events = max_events
        self.max_jets = max_jets
        self.num_workers = int(num_workers)
        self.worker_prefetch = int(worker_prefetch)
        self.multiprocessing_start_method = multiprocessing_start_method
        self.verbose = verbose
        self.means, self.stds = self._normalization_tensors(normalizer)

    def _normalization_tensors(self, normalizer):
        if normalizer is None:
            return None, None
        if isinstance(normalizer, dict):
            means = normalizer["means"]
            stds = normalizer["stds"]
        elif hasattr(normalizer, "means") and hasattr(normalizer, "stds"):
            means = normalizer.means
            stds = normalizer.stds
        else:
            raise TypeError(
                "normalizer must be a mapping or expose means and stds"
            )

        means = torch.as_tensor(
            means, dtype=torch.float, device=self.device
        )
        stds = torch.as_tensor(
            stds, dtype=torch.float, device=self.device
        )
        if means.ndim != 1 or stds.shape != means.shape:
            raise ValueError("normalization means and stds must be 1D matches")
        if not torch.isfinite(means).all() or not torch.isfinite(stds).all():
            raise ValueError("normalization parameters must be finite")
        if (stds == 0).any():
            raise ValueError("normalization standard deviations cannot be zero")
        return means, stds

    def _score_graphs(self, graphs):
        batch = Batch.from_data_list(graphs).to(self.device)
        x = batch.x
        if self.means is not None:
            if x.shape[1] != len(self.means):
                raise ValueError(
                    f"Graph has {x.shape[1]} features, but normalizer has "
                    f"{len(self.means)}"
                )
            x = (x - self.means) / self.stds

        with torch.inference_mode():
            total, _, _ = self.model.score_batch(
                x,
                batch.edge_index,
                batch.batch,
                alpha=self.alpha,
            )
        return total.detach().cpu().tolist()

    def _assemble_scores(self, jet_counts, graphs, positions):
        # NaN preserves the original jet position if graph construction
        # rejects an empty or otherwise unusable jet.
        event_scores = [[np.nan] * count for count in jet_counts]
        for start in range(0, len(graphs), self.graph_batch_size):
            stop = start + self.graph_batch_size
            scores = self._score_graphs(graphs[start:stop])
            for (event_index, jet_index), score in zip(
                positions[start:stop], scores
            ):
                event_scores[event_index][jet_index] = score
        return ak.Array(event_scores)

    def _entry_ranges(self, entry_stop):
        for entry_start in range(
            0, entry_stop, self.root_entry_batch_size
        ):
            yield (
                entry_start,
                min(entry_start + self.root_entry_batch_size, entry_stop),
            )

    def _iter_sequential_ranges(self, file_name, branches, entry_stop):
        with uproot.open(f"{file_name}:Events") as tree:
            for entry_start, batch_stop in self._entry_ranges(entry_stop):
                events = tree.arrays(
                    branches,
                    entry_start=entry_start,
                    entry_stop=batch_stop,
                    library="ak",
                )
                yield (
                    entry_start,
                    batch_stop,
                    _build_graphs(
                        events, self.graph_config, self.max_jets
                    ),
                )

    def _worker_context(self):
        method = self.multiprocessing_start_method
        if method is None:
            # Do not fork a process after PyTorch or CUDA has initialized
            # threads. Spawn is the portable safe default.
            method = "spawn"
        return multiprocessing.get_context(method)

    def _iter_parallel_ranges(self, file_name, branches, entry_stop):
        ranges = iter(self._entry_ranges(entry_stop))
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=self.num_workers,
            mp_context=self._worker_context(),
            initializer=_initialize_scorer_worker,
            initargs=(
                self.graph_config,
                self.max_jets,
                file_name,
                branches,
            ),
        ) as executor:
            pending = collections.deque()
            max_pending = self.num_workers * self.worker_prefetch

            def submit_one():
                try:
                    entry_start, batch_stop = next(ranges)
                except StopIteration:
                    return False
                task = (entry_start, batch_stop)
                pending.append(
                    (entry_start, batch_stop, executor.submit(
                        _process_scorer_entry_range, task
                    ))
                )
                return True

            for _ in range(max_pending):
                if not submit_one():
                    break

            while pending:
                entry_start, batch_stop, future = pending.popleft()
                result = pickle.loads(future.result())
                yield entry_start, batch_stop, result
                submit_one()

    def iter_file(self, file_name) -> Iterator[ScoreChunk]:
        """Yield ordered score chunks for *file_name*."""
        # pathlib.Path turns root:// into root:/ and breaks xrootd access.
        file_name = normalize_input_path(file_name)
        branches = tuple(pfnano_branches(self.graph_config))
        with uproot.open(f"{file_name}:Events") as tree:
            entry_stop = tree.num_entries
        if self.max_events is not None:
            entry_stop = min(entry_stop, self.max_events)

        ranges = (
            self._iter_sequential_ranges(file_name, branches, entry_stop)
            if self.num_workers == 1
            else self._iter_parallel_ranges(file_name, branches, entry_stop)
        )
        progress = tqdm(
            total=entry_stop,
            desc=f"Scoring {os.path.basename(file_name)}",
            disable=not self.verbose,
        )
        try:
            for entry_start, batch_stop, result in ranges:
                jet_counts, graphs, positions = result
                scores = self._assemble_scores(
                    jet_counts, graphs, positions
                )
                progress.update(batch_stop - entry_start)
                yield ScoreChunk(
                    file_name=file_name,
                    entry_start=entry_start,
                    entry_stop=batch_stop,
                    scores=scores,
                )
        finally:
            if hasattr(ranges, "close"):
                ranges.close()
            progress.close()

    def score_file(self, file_name):
        """Return one event-ordered jagged score array for *file_name*."""
        chunks = [chunk.scores for chunk in self.iter_file(file_name)]
        return ak.concatenate(chunks) if chunks else ak.Array([])

    def score_files(self, file_names: Sequence[str]):
        """Return score arrays keyed by input filename, preserving file order."""
        if isinstance(file_names, (str, Path)):
            file_names = [file_names]
        return OrderedDict(
            (str(file_name), self.score_file(file_name))
            for file_name in file_names
        )
