import collections
import concurrent.futures
import hashlib
import multiprocessing
import os
import pickle
import re
import sys
import warnings
import numpy as np
import uproot
import torch
from torch_geometric.data import Dataset, Data
try:
    from torch_geometric.data.dataset import _repr, files_exist
except ImportError:
    def _repr(obj):
        return 'None' if obj is None else re.sub(r'(<.*?)\s.*(>)', r'\1\2', str(obj))
    def files_exist(files):
        return len(files) != 0 and all(os.path.exists(f) for f in files)
from tqdm import tqdm
import logging
from utils.GraphBuilder import (
    GraphBuildConfig,
    build_pfnano_jet_graph,
    pfnano_branches,
)
from utils.Cuts import (
    cut_branches,
    cuts_fingerprint,
    event_cut_mask,
    jet_cut_mask,
)
from utils.InputJson import dataset_of, sum_initial_events


_GRAPH_WORKER_PRODUCER = None


def _initialize_graph_worker(producer):
    """Install the producer once per worker and avoid nested CPU pools."""
    global _GRAPH_WORKER_PRODUCER
    _GRAPH_WORKER_PRODUCER = producer
    _GRAPH_WORKER_PRODUCER._chunk_cache = collections.OrderedDict()
    torch.set_num_threads(1)


def _process_graph_entry_range(task):
    """Worker entry point; task results are consumed in submission order."""
    if _GRAPH_WORKER_PRODUCER is None:
        raise RuntimeError("Graph worker was not initialized")
    try:
        graphs = _GRAPH_WORKER_PRODUCER._process_entry_range(*task)
        # Returning tensors directly makes multiprocessing use torch's shared
        # memory reducer. Explicit bytes work in containers without
        # torch_shm_manager and avoid exhausting /dev/shm on large batches.
        return pickle.dumps(graphs, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as exc:
        file_name, _, _, entry_start, entry_stop = task[:5]
        raise RuntimeError(
            f"Failed to process entries [{entry_start}, {entry_stop}) "
            f"from '{file_name}'"
        ) from exc


class JetGraphProducer(Dataset):
    _CONFIG_VERSION = 1

    """
    Produces graphs from a root file containing jet constituents.
    The graphs are built by reclustering the constituents with the Cambridge-Aachen algorithm.
    Supports both spacial graphs and lund graphs.
    For spacial graphs, graph nodes are the constituents, while the edges encode vicinity in a given metric.
    For Lund graphs, graph nodes are pseudojets, while the edges encode the clustering history.
    For spacial graphs, the node features are the pt, eta, phi, pdgId and mass of the constituents.
    For Lund graphs, the node features are the pt, eta, phi, mass, Lund coordinates and energy fractions of the pseudojets.
    Spacial graphs can optionally have edge features, which are the invariant mass, kt distance and CA distance between constituent pairs.
    The graphs are stored in a pytorch geometric dataset and written to disk in a "processed" folder in the root file directory.
    
    Args:
        root (str): Path to the folder with (one or more) root files containing the jet constituents. All files in the folder will be processed and treated as one sample.
        output_dataset_name (str): Name of the processed output dataset
        n_store_jets (int): Number of jets to store per event
        delta_r_threshold (float): Threshold for the deltaR distance between constituents to be considered connected (spacial graphs only)
        n_store_cands (int): Number of constituents to store per jet (spacial graphs only)
        max_events_to_process (int): Maximum number of events to process
        use_delta_r_star (bool): Whether to use the deltaR* distance instead of the standard deltaR (TODO: implement)
        use_delta_r_star_star (bool): Whether to use the deltaR** distance instead of the standard deltaR (TODO: implement)
        transform (callable, optional): A function/transform that takes in an :obj:`torch_geometric.data.Data` object and returns a transformed version. The data object will be transformed before every access. (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in an :obj:`torch_geometric.data.Data` object and returns a transformed version. The data object will be transformed before being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an :obj:`torch_geometric.data.Data` object and returns a boolean value, indicating whether the data object should be included in the final dataset. (default: :obj:`None`)
        use_relative_angles (bool): Whether to use relative angles (eta - jet_eta, phi - jet_phi) or absolute angles (eta, phi)
        use_dummy_values (bool): Whether to use dummy values for the features instead of the actual values
        save_edge_attributes (bool): Whether to save the edge attributes (pairwise invariant masses, Cambridge-Aachen and kt distances)
        save_n_constituents (bool): Whether to save the number of constituents in the graph
        save_event_number (bool): Whether to save the event number in the graph
        use_lund_decomp (bool): Whether to use the Lund decomposition instead of the spacial graph construction
        n_lund_vars (int): Number of Lund variables to use (3 or 5)
        save_4vectors_in_lund_tree (bool): Whether to save the 4-vectors of the constituents in the Lund tree
        kt_cut (float): kt cut to prune the tree (if None, no pruning is performed)
        extra_label (str): Extra label to add to the processed file name)
        weights (str): What event weights to use (None, 'xsec') (TODO: implement pt flattening)
        extra_obs_to_save_per_jet (list): List of extra jet-level observables to store in the graph (must be available in the root file)
        extra_obs_to_save_per_event (list): List of extra event-level observables to store in the graph (must be available in the root file)
        extra_obs_to_compute_per_event (list): List of extra event-level observables to compute and store in the graph (callable functions that take the event as input and return a float)
        extra_obs_to_load (list): List of extra observables to load from the root file (must be available in the root file)
        save_jet_idx (bool): Whether to save the jet index
        label (float): Label to add to the graph for supervised learning
        verbose (bool): Whether to print progress bars
        input_format (str): Input format. Only ``"PFNanoAOD"`` is supported.
        jet_collection (str): Name of the jet collection in the root file (default: "FatJet")
        use_displacement (bool): Whether to use the displacement features (d0, dz) in the Lund decomposition (Scouting Data has no displacement information)
        graphs_per_chunk (int): Number of graphs stored per output chunk (default: 1000).
            Larger values reduce the number of files but increase per-chunk memory use.
        n_cached_chunks (int): Number of chunk files to keep in memory at once (default: 2).
            Increase for random-access workloads; decrease to save RAM.
        num_workers (int): Number of graph-production worker processes. Values
            greater than one enable multiprocessing (default: 1).
        event_batch_size (int): Events assigned to each worker task.
            Defaults to ``graphs_per_chunk``.
        worker_prefetch (int): Maximum queued tasks per worker (default: 2).

    Note:
        Processed datasets are stored as a metadata file ``{name}_meta.pt`` and a set of
        chunk files ``{name}_chunk_{i}.pt`` inside the ``processed/`` sub-directory.
        Datasets processed with the old ``InMemoryDataset``-based format (a single
        ``{name}.pt`` file) must be re-processed before they can be used with this class.
    """

    def __init__(
        self,
        root,
        output_dataset_name,
        n_store_jets=2,
        delta_r_threshold=0.2,
        n_store_cands=None,
        max_events_to_process=None,
        use_delta_r_star=False,
        use_delta_r_star_star=False,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        use_relative_angles=True,
        use_dummy_values=False,
        save_edge_attributes=False,
        save_n_constituents=False,
        save_event_number=False,
        use_lund_decomp=False,
        n_lund_vars=3,
        save_4vectors_in_lund_tree=False,
        save_energy_fractions=True,
        kt_cut=None,
        weights=None,
        extra_obs_to_save_per_jet=None,
        extra_obs_to_save_per_event=None,
        extra_obs_to_compute_per_event=None,
        extra_obs_to_load=None,
        save_jet_idx=False,
        label:float=0.,
        verbose=False,
        input_format="PFNanoAOD",
        jet_collection="FatJet",
        use_displacement=False,
        graphs_per_chunk=None,
        event_batch_size=None,
        n_cached_chunks=2,
        validate_cache_config=True,
        num_workers=1,
        # Deprecated aliases retained for existing callers.
        chunk_size=None,
        processing_batch_size=None,
        worker_prefetch=2,
        multiprocessing_start_method=None,
    ):

        self.multiprocessing_start_method = multiprocessing_start_method

        self.root = root
        self.n_store_jets = n_store_jets
        self.output_dataset_name = output_dataset_name
        self.n_store_cands = n_store_cands
        self.delta_r_threshold = delta_r_threshold
        self.max_events_to_process = max_events_to_process
        self.use_delta_r_star = use_delta_r_star
        self.use_delta_r_star_star = use_delta_r_star_star
        self.pre_transform = pre_transform
        self.use_relative_angles = use_relative_angles
        self.use_dummy_values = use_dummy_values
        self.save_edge_attributes = save_edge_attributes
        self.save_n_constituents = save_n_constituents
        self.save_event_number = save_event_number
        self.use_lund_decomp = use_lund_decomp
        self.n_lund_vars = n_lund_vars
        self.save_4vectors_in_lund_tree = save_4vectors_in_lund_tree
        self.save_energy_fractions = save_energy_fractions
        self.kt_cut = kt_cut
        self.weights = weights
        self.extra_obs_to_save_per_jet = list(extra_obs_to_save_per_jet or [])
        self.extra_obs_to_save_per_event = list(extra_obs_to_save_per_event or [])
        self.extra_obs_to_compute_per_event = list(extra_obs_to_compute_per_event or [])
        self.extra_obs_to_load = list(extra_obs_to_load or [])
        self.save_jet_idx = save_jet_idx
        self.label = label
        self.verbose = verbose
        self.input_format = input_format
        self.jet_collection = jet_collection
        self.use_displacement = use_displacement
        if graphs_per_chunk is None:
            graphs_per_chunk = 1000 if chunk_size is None else chunk_size
        elif chunk_size is not None and chunk_size != graphs_per_chunk:
            raise ValueError(
                "graphs_per_chunk and deprecated chunk_size disagree"
            )
        if event_batch_size is None:
            event_batch_size = (
                processing_batch_size
                if processing_batch_size is not None
                else graphs_per_chunk
            )
        elif (
            processing_batch_size is not None
            and processing_batch_size != event_batch_size
        ):
            raise ValueError(
                "event_batch_size and deprecated processing_batch_size disagree"
            )
        self.graphs_per_chunk = int(graphs_per_chunk)
        self.event_batch_size = int(event_batch_size)
        # Compatibility aliases for external code.
        self.chunk_size = self.graphs_per_chunk
        self.processing_batch_size = self.event_batch_size
        cache_env = os.environ.get("GAE_CHUNK_CACHE_SIZE")
        self._n_cached_chunks = int(cache_env) if cache_env else (50 if n_cached_chunks == 2 else n_cached_chunks)
        self.validate_cache_config = validate_cache_config
        self.num_workers = int(num_workers)
        self.worker_prefetch = int(worker_prefetch)
        # OrderedDict used as a bounded LRU cache: key = chunk index, value = list of graphs
        self._chunk_cache: collections.OrderedDict = collections.OrderedDict()

        assert self.weights in ["xsec", None], "Only xsec and None are supported for the weights argument (more to be added later)"
        if self.num_workers < 1:
            raise ValueError("num_workers must be at least 1")
        if self.graphs_per_chunk < 1:
            raise ValueError("graphs_per_chunk must be at least 1")
        if self.event_batch_size < 1:
            raise ValueError("event_batch_size must be at least 1")
        if self.worker_prefetch < 1:
            raise ValueError("worker_prefetch must be at least 1")

        if input_format != "PFNanoAOD":
            raise ValueError(
                "Only input_format='PFNanoAOD' is supported; "
                "TreeMaker support has been removed"
            )

        # Assert that no incompatible flags are used
        if use_lund_decomp:
            assert use_delta_r_star == False, "use_delta_r_star is not compatible with use_lund_decomp"
            assert use_delta_r_star_star == False, "use_delta_r_star_star is not compatible with use_lund_decomp"
            assert use_relative_angles == True, "use_relative_angles is not compatible with use_lund_decomp"
            assert use_dummy_values == False, "use_dummy_values is not compatible with use_lund_decomp"
            assert save_edge_attributes == False, "save_edge_attributes is not compatible with use_lund_decomp"
        
        else:
            assert kt_cut is None, "kt_cut is only supported for Lund tree representations"

        # If xsec weighs are requested but the number of events is hard capped, warn that an appoximation is used
        if self.weights == "xsec" and self.max_events_to_process:
            logging.warning("Using approximation for initial number of events with max number of events")

        super().__init__(root, transform, pre_transform, pre_filter)

        # Load lightweight metadata from disk (weights, jet indices, total count).
        # No graph tensors are loaded here — those are fetched on-demand in get().
        meta_path = os.path.join(self.processed_dir, f"{self.output_dataset_name}_meta.pt")
        meta = torch.load(meta_path, map_location="cpu", weights_only=False)
        if self.validate_cache_config:
            self._validate_processing_config(meta, meta_path)
        # The on-disk layout must always come from metadata. This is
        # particularly important for read-only loaders that intentionally do
        # not reconstruct all producer arguments.
        self.graphs_per_chunk = int(
            meta.get("graphs_per_chunk", meta["chunk_size"])
        )
        self.chunk_size = self.graphs_per_chunk
        self._n_graphs: int = meta["n_graphs"]
        self._w: torch.Tensor = meta["w"]
        self._jet_idx_all: torch.Tensor = meta.get("jet_idx")

    @staticmethod
    def _callable_name(value):
        """Return a stable, human-readable identifier for a callable."""
        if value is None:
            return None
        module = getattr(value, "__module__", value.__class__.__module__)
        name = getattr(
            value,
            "__qualname__",
            getattr(value, "__name__", value.__class__.__qualname__),
        )
        return f"{module}.{name}"

    def _processing_config(self):
        """Configuration fields that affect the contents/layout of cached graphs."""
        return {
            "config_version": self._CONFIG_VERSION,
            "n_store_jets": self.n_store_jets,
            "n_store_cands": self.n_store_cands,
            "delta_r_threshold": self.delta_r_threshold,
            "max_events_to_process": self.max_events_to_process,
            "use_delta_r_star": self.use_delta_r_star,
            "use_delta_r_star_star": self.use_delta_r_star_star,
            "pre_transform": self._callable_name(self.pre_transform),
            "pre_filter": self._callable_name(self.pre_filter),
            "use_relative_angles": self.use_relative_angles,
            "use_dummy_values": self.use_dummy_values,
            "save_edge_attributes": self.save_edge_attributes,
            "save_n_constituents": self.save_n_constituents,
            "save_event_number": self.save_event_number,
            "use_lund_decomp": self.use_lund_decomp,
            "n_lund_vars": self.n_lund_vars,
            "save_4vectors_in_lund_tree": self.save_4vectors_in_lund_tree,
            "save_energy_fractions": self.save_energy_fractions,
            "kt_cut": self.kt_cut,
            "weights": self.weights,
            "extra_obs_to_save_per_jet": list(self.extra_obs_to_save_per_jet),
            "extra_obs_to_save_per_event": list(self.extra_obs_to_save_per_event),
            "extra_obs_to_compute_per_event": [
                self._callable_name(value)
                for value in self.extra_obs_to_compute_per_event
            ],
            "extra_obs_to_load": list(self.extra_obs_to_load),
            "save_jet_idx": self.save_jet_idx,
            # Retained as None for compatibility with PFNanoAOD caches written
            # before obsolete TreeMaker mask support was removed.
            "mask_sha256": None,
            "label": self.label,
            "input_format": self.input_format,
            "jet_collection": self.jet_collection,
            "use_displacement": self.use_displacement,
            # get() uses this value to map global indices to chunk files.
            "chunk_size": self.graphs_per_chunk,
        }

    def _validate_processing_config(self, meta, meta_path):
        saved = meta.get("processing_config")
        requested = self._processing_config()

        if saved is None:
            raise RuntimeError(
                f"Cached dataset '{self.output_dataset_name}' at '{meta_path}' "
                "was created without producer configuration metadata, so it "
                "cannot be safely reused. Choose a new output_dataset_name or "
                "remove this dataset's metadata and chunk files to reprocess it."
            )

        differing = {
            key: (saved.get(key, "<missing>"), requested.get(key, "<missing>"))
            for key in sorted(set(saved) | set(requested))
            if saved.get(key, "<missing>") != requested.get(key, "<missing>")
        }
        if differing:
            details = "\n".join(
                f"  {key}: cached={old!r}, requested={new!r}"
                for key, (old, new) in differing.items()
            )
            raise RuntimeError(
                f"Cached dataset '{self.output_dataset_name}' was produced with "
                f"incompatible settings:\n{details}\n"
                "Choose a new output_dataset_name or remove this dataset's "
                "metadata and chunk files to reprocess it."
            )

    # ------------------------------------------------------------------
    # Properties that return per-sample values respecting the current
    # index view (set by Dataset.index_select / slicing / permutation).
    # ------------------------------------------------------------------

    @property
    def w(self) -> torch.Tensor:
        """Event weights for the graphs in the current dataset view."""
        if self._indices is None:
            return self._w
        return self._w[self._indices]

    @property
    def jet_idx(self) -> torch.Tensor:
        """Jet indices for the graphs in the current dataset view."""
        if self._jet_idx_all is None:
            raise AttributeError("jet_idx was not saved during processing (save_jet_idx=False)")
        if self._indices is None:
            return self._jet_idx_all
        return self._jet_idx_all[self._indices]

    # ------------------------------------------------------------------
    # torch_geometric.data.Dataset interface
    # ------------------------------------------------------------------

    def len(self) -> int:
        return self._n_graphs

    @property
    def raw_file_names(self):
        return [
            os.path.join(self.root, name)
            for name in sorted(os.listdir(self.root))
            if name.endswith(".root")
        ]

    @property
    def processed_file_names(self):
        # Only the metadata file is used to determine whether processing has been done.
        # The chunk files are written alongside it during process() and are discovered
        # at load time via get().
        return [f"{self.output_dataset_name}_meta.pt"]
    
    def get_total_initial_events(self):
        n_initial = 0
        for file_name in self.raw_file_names:
            with uproot.open(file_name) as input_file:
                if "CutFlow" not in input_file:
                    raise ValueError("No CutFlow tree found in file")
                n_initial += input_file["CutFlow"]["Initial"].array(
                    entry_stop=1
                )[0]
        return n_initial

    def _graph_build_config(self):
        return GraphBuildConfig(
            jet_collection=self.jet_collection,
            n_store_cands=self.n_store_cands,
            use_relative_angles=self.use_relative_angles,
            use_displacement=self.use_displacement,
            use_dummy_values=self.use_dummy_values,
            use_lund_decomp=self.use_lund_decomp,
            delta_r_threshold=self.delta_r_threshold,
            save_edge_attributes=self.save_edge_attributes,
            n_lund_vars=self.n_lund_vars,
            save_4vectors_in_lund_tree=self.save_4vectors_in_lund_tree,
            save_energy_fractions=self.save_energy_fractions,
            kt_cut=self.kt_cut,
            label=self.label,
        )

    def process_event(
        self,
        i_start,
        i_stop,
        events,
        xsec,
        gen_events_before_selection,
        total_events,
        events_to_process,
        show_progress=None,
    ):
        graphs = []
        show_progress = self.verbose if show_progress is None else show_progress
        config = self._graph_build_config()

        for i_ev in tqdm(range(i_start, i_stop), disable=not show_progress):
            event = events[i_ev]
            n_jets = min(
                self.n_store_jets,
                int(event[f"n{self.jet_collection}"]),
            )
            for jet_index in range(n_jets):
                graph, n_constituents = build_pfnano_jet_graph(
                    event, jet_index, config
                )
                if graph is None:
                    logging.warning(
                        "Skipping event %s jet %s because it has no constituents",
                        i_ev,
                        jet_index,
                    )
                    continue

                if self.weights == "xsec":
                    if not self.max_events_to_process:
                        weight = xsec / gen_events_before_selection
                    else:
                        weight = xsec / (
                            gen_events_before_selection
                            * events_to_process
                            / total_events
                        )
                    graph.w = torch.tensor([weight], dtype=torch.float)

                if self.save_n_constituents:
                    graph.n_constituents = torch.tensor(
                        [n_constituents], dtype=torch.long
                    )
                if self.save_event_number:
                    graph.event_number = torch.tensor([i_ev], dtype=torch.long)
                if self.save_jet_idx:
                    graph.jet_idx = torch.tensor([jet_index], dtype=torch.int)

                for observable in self.extra_obs_to_save_per_jet:
                    graph[observable] = torch.tensor(
                        [event[observable][jet_index]], dtype=torch.float
                    )
                for observable in self.extra_obs_to_save_per_event:
                    graph[observable] = torch.tensor(
                        [event[observable]], dtype=torch.float
                    )
                for observable in self.extra_obs_to_compute_per_event:
                    graph[observable.__name__] = torch.tensor(
                        [observable(event)], dtype=torch.float
                    )
                graphs.append(graph)

        return graphs
    def get_graphs(self, file_name):
        xsec = 1.0
        gen_events_before_selection = None
        if self.weights == "xsec":
            with uproot.open(file_name) as input_file:
                if "Metadata" not in input_file:
                    raise ValueError("No Metadata tree found in file")
                if "GenCrossSection" not in input_file["Metadata"].keys():
                    raise ValueError("No GenCrossSection found in Metadata")
                xsec = input_file["Metadata"]["GenCrossSection"].array(
                    entry_stop=1
                )[0]
                gen_events_before_selection = self.get_total_initial_events()

        extra_branches = (
            self.extra_obs_to_load
            + self.extra_obs_to_save_per_event
            + self.extra_obs_to_save_per_jet
        )
        branches = pfnano_branches(
            self._graph_build_config(), extra_branches
        )
        with uproot.open(f"{file_name}:Events") as tree:
            total_events = tree.num_entries
            events_to_process = (
                min(self.max_events_to_process, total_events)
                if self.max_events_to_process
                else total_events
            )
            events = tree.arrays(
                branches, entry_stop=events_to_process, library="ak"
            )

        if gen_events_before_selection is None:
            gen_events_before_selection = total_events
        graphs = self.process_event(
            0,
            events_to_process,
            events,
            xsec,
            gen_events_before_selection,
            total_events,
            events_to_process,
        )
        if self.pre_transform is not None:
            graphs = [self.pre_transform(graph) for graph in graphs]
        return graphs
    def _process_entry_range(
        self,
        file_name,
        tree_path,
        branches_to_load,
        entry_start,
        entry_stop,
        xsec,
        gen_events_before_selection,
        total_events,
        events_to_process,
    ):
        """Read and process one independent range of ROOT entries."""
        with uproot.open(f"{file_name}:{tree_path}") as tree:
            events_batch = tree.arrays(
                branches_to_load,
                entry_start=entry_start,
                entry_stop=entry_stop,
                library="ak",
            )

        graphs = self.process_event(
            0,
            len(events_batch),
            events_batch,
            xsec,
            gen_events_before_selection,
            total_events,
            events_to_process,
            show_progress=False,
        )
        if self.save_event_number:
            for graph in graphs:
                graph.event_number = graph.event_number + entry_start
        if self.pre_transform is not None:
            graphs = [self.pre_transform(graph) for graph in graphs]
        if self.pre_filter is not None:
            graphs = [graph for graph in graphs if self.pre_filter(graph)]
        return graphs

    def _iter_entry_ranges(
        self,
        executor,
        file_name,
        tree_path,
        branches_to_load,
        total_events,
        events_to_process,
        xsec,
        gen_events_before_selection,
    ):
        tasks = (
            (
                file_name,
                tree_path,
                branches_to_load,
                entry_start,
                min(entry_start + self.event_batch_size, events_to_process),
                xsec,
                gen_events_before_selection,
                total_events,
                events_to_process,
            )
            for entry_start in range(
                0, events_to_process, self.event_batch_size
            )
        )

        with tqdm(
            total=events_to_process,
            desc=f"Processing {os.path.basename(file_name)}",
            disable=not self.verbose,
        ) as progress:
            if executor is None:
                for task in tasks:
                    graphs = self._process_entry_range(*task)
                    progress.update(task[4] - task[3])
                    yield from graphs
                return

            # Keep only a bounded number of ranges in flight. Futures are
            # consumed in submission order, preserving graph ordering.
            pending = collections.deque()
            max_pending = self.num_workers * self.worker_prefetch
            for _ in range(max_pending):
                try:
                    task = next(tasks)
                    pending.append((
                        executor.submit(_process_graph_entry_range, task),
                        task[4] - task[3],
                    ))
                except StopIteration:
                    break

            while pending:
                future, n_entries = pending.popleft()
                graphs = pickle.loads(future.result())
                progress.update(n_entries)
                yield from graphs
                try:
                    task = next(tasks)
                    pending.append((
                        executor.submit(_process_graph_entry_range, task),
                        task[4] - task[3],
                    ))
                except StopIteration:
                    pass

    def _iter_graphs_from_file(self, file_name, executor=None):
        """Yield PFNanoAOD graphs from deterministic event ranges."""
        xsec = 1.0
        gen_events_before_selection = None
        if self.weights == "xsec":
            with uproot.open(file_name) as metadata_file:
                if "Metadata" not in metadata_file:
                    raise ValueError("No Metadata tree found in file")
                if "GenCrossSection" not in metadata_file["Metadata"].keys():
                    raise ValueError("No GenCrossSection found in Metadata")
                xsec = metadata_file["Metadata"]["GenCrossSection"].array(
                    entry_stop=1
                )[0]
                gen_events_before_selection = self.get_total_initial_events()

        extra_branches = (
            self.extra_obs_to_load
            + self.extra_obs_to_save_per_event
            + self.extra_obs_to_save_per_jet
        )
        branches = pfnano_branches(
            self._graph_build_config(), extra_branches
        )
        with uproot.open(f"{file_name}:Events") as tree:
            total_events = tree.num_entries

        if gen_events_before_selection is None:
            gen_events_before_selection = total_events
        events_to_process = (
            min(self.max_events_to_process, total_events)
            if self.max_events_to_process
            else total_events
        )
        yield from self._iter_entry_ranges(
            executor,
            file_name,
            "Events",
            branches,
            total_events,
            events_to_process,
            xsec,
            gen_events_before_selection,
        )
    def get(self, idx: int) -> Data:
        """Load and return the graph at global position *idx*.

        Graphs are stored on disk in fixed-size chunk files.  The relevant
        chunk is loaded into a small in-memory LRU cache so that sequential
        and batch-level access patterns reuse already-loaded chunks without
        keeping the entire dataset in RAM.
        """
        chunk_idx = idx // self.graphs_per_chunk
        pos = idx % self.graphs_per_chunk

        if chunk_idx not in self._chunk_cache:
            chunk_path = os.path.join(
                self.processed_dir,
                f"{self.output_dataset_name}_chunk_{chunk_idx}.pt",
            )
            chunk = torch.load(chunk_path, map_location="cpu", weights_only=False)
            # Evict the oldest entry when the cache is full
            if len(self._chunk_cache) >= self._n_cached_chunks:
                self._chunk_cache.popitem(last=False)
            self._chunk_cache[chunk_idx] = chunk
        else:
            # Move to the end to mark as most-recently used
            self._chunk_cache.move_to_end(chunk_idx)

        # Return a clone so that in-place transforms applied by __getitem__
        # (e.g. OnTheFlyNormalizer) do not corrupt the cached graph objects.
        # Without this, the second access of the same graph would receive an
        # already-transformed tensor and produce inf/nan values.
        return self._chunk_cache[chunk_idx][pos].clone()

    def process(self):
        # Stream graphs from every input file and write them to disk in
        # chunk files as they are produced, so that the peak RAM usage
        # during processing is proportional to graphs_per_chunk, not to the
        # total dataset size.
        buffer = []
        chunk_idx = 0
        n_graphs = 0
        w_list = []
        jet_idx_list = [] if self.save_jet_idx else None
        default_weight = torch.tensor([1.0])

        def _flush_chunk(data, idx):
            chunk_path = os.path.join(
                self.processed_dir,
                f"{self.output_dataset_name}_chunk_{idx}.pt",
            )
            torch.save(data, chunk_path)

        def _consume_files(executor=None):
            nonlocal buffer, chunk_idx, n_graphs
            for file in self.raw_file_names:
                print(f"Processing {file}")
                for graph in self._iter_graphs_from_file(file, executor):
                    buffer.append(graph)
                    w_list.append(
                        graph.w if hasattr(graph, "w") else default_weight
                    )
                    if jet_idx_list is not None:
                        jet_idx_list.append(graph.jet_idx)
                    n_graphs += 1

                    if len(buffer) == self.graphs_per_chunk:
                        _flush_chunk(buffer, chunk_idx)
                        buffer = []
                        chunk_idx += 1

        if self.num_workers == 1:
            _consume_files()
        else:
            method = self.multiprocessing_start_method
            if method is None:
                # Do not fork after PyTorch/ROOT has initialized threads.
                # Forking in a multi-threaded process causes deadlocks.
                start_methods = multiprocessing.get_all_start_methods()
                method = "spawn" if "spawn" in start_methods else None
            context = multiprocessing.get_context(method)
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=self.num_workers,
                mp_context=context,
                initializer=_initialize_graph_worker,
                initargs=(self,),
            ) as executor:
                _consume_files(executor)

        # Flush any graphs that didn't fill a complete chunk
        if buffer:
            _flush_chunk(buffer, chunk_idx)
            chunk_idx += 1

        meta = {
            "n_graphs": n_graphs,
            "n_chunks": chunk_idx,
            "chunk_size": self.graphs_per_chunk,
            "graphs_per_chunk": self.graphs_per_chunk,
            "w": torch.cat(w_list) if w_list else torch.empty(0),
            "jet_idx": (
                torch.cat(jet_idx_list)
                if jet_idx_list
                else None
            ),
            "processing_config": self._processing_config(),
        }
        torch.save(
            meta,
            os.path.join(self.processed_dir, f"{self.output_dataset_name}_meta.pt"),
        )

    @property
    def pre_transform_file(self) -> str:
        """Path to the shard-specific pre_transform marker file."""
        return os.path.join(self.processed_dir, f"{self.output_dataset_name}_pre_transform.pt")

    @property
    def pre_filter_file(self) -> str:
        """Path to the shard-specific pre_filter marker file."""
        return os.path.join(self.processed_dir, f"{self.output_dataset_name}_pre_filter.pt")

    def _process(self):
        """Process graphs and save transform/filter markers with shard names to prevent write collisions."""
        for file_path, legacy_name, current_val, name in [
            (self.pre_transform_file, "pre_transform.pt", self.pre_transform, "pre_transform"),
            (self.pre_filter_file, "pre_filter.pt", self.pre_filter, "pre_filter"),
        ]:
            check_path = file_path if os.path.exists(file_path) else os.path.join(self.processed_dir, legacy_name)
            if os.path.exists(check_path):
                try:
                    saved_val = torch.load(check_path, map_location="cpu", weights_only=False)
                    if saved_val != _repr(current_val):
                        warnings.warn(
                            f"The `{name}` argument differs from the one used in "
                            "the pre-processed version of this dataset. If you want to "
                            "make use of another pre-processing technique, pass "
                            "`force_reload=True` explicitly to reload the dataset."
                        )
                except Exception:
                    pass

        if not getattr(self, "force_reload", False) and files_exist(self.processed_paths):
            return

        if getattr(self, "log", True) and "pytest" not in sys.modules:
            print("Processing...", file=sys.stderr)

        os.makedirs(self.processed_dir, exist_ok=True)
        self.process()

        torch.save(_repr(self.pre_transform), self.pre_transform_file)
        torch.save(_repr(self.pre_filter), self.pre_filter_file)

        for legacy_name, obj in [
            ("pre_transform.pt", self.pre_transform),
            ("pre_filter.pt", self.pre_filter),
        ]:
            legacy_path = os.path.join(self.processed_dir, legacy_name)
            try:
                if not os.path.exists(legacy_path):
                    torch.save(_repr(obj), legacy_path)
            except OSError:
                pass

        if getattr(self, "log", True) and "pytest" not in sys.modules:
            print("Done!", file=sys.stderr)


class FileListGraphProducer(JetGraphProducer):
    """Produce graphs from an explicit list of ROOT files, with selections.

    Differs from :class:`JetGraphProducer` in three ways:

    - the input files are given explicitly (typically read from the JSON in
      ``inputJson/``) instead of every ``.root`` file inside ``root``.
      ``root`` is therefore only the output location of the ``processed/``
      directory, which makes it straightforward to give each condor or crab
      shard its own output area;
    - the selections in :mod:`utils.Cuts` are applied to every event and jet
      according to the sample's ``role`` and ``year``;
    - with ``weights="xsec"`` the number of generated events is summed over
      the files of the dataset a file belongs to (``dataset_groups``), not
      over every input file, so several datasets can share one sample. The
      sums are cached so each file is opened once for this purpose.

    ``save_jet_idx`` is forced on because the jet index is needed to evaluate
    the per-jet selection, and the cut branches are appended to
    ``extra_obs_to_load`` so the cut variables are read from the file.

    Args:
        root (str): Output directory for the processed dataset.
        output_dataset_name (str): Name of the processed output dataset.
        file_paths (list): ROOT paths to process.
        role (str): ``"signal"``, ``"background"`` or ``"data"``.
        year (str): Data-taking year the sample corresponds to.
        dataset_groups (dict): ``{dataset: [files]}`` covering
            ``file_paths``. Defaults to a single group holding every file.

    All remaining arguments are passed through to :class:`JetGraphProducer`.
    """

    def __init__(
        self,
        root,
        output_dataset_name,
        file_paths,
        role,
        year,
        dataset_groups=None,
        **kwargs,
    ):
        self._file_paths = [str(path) for path in file_paths]
        if not self._file_paths:
            raise ValueError("No input ROOT files were provided")
        self.role = role
        self.year = str(year)
        self.region = kwargs.pop("region", "control")
        if dataset_groups is None:
            dataset_groups = {output_dataset_name: list(self._file_paths)}
        self.dataset_groups = {
            name: [str(path) for path in files]
            for name, files in dataset_groups.items()
        }
        self._initial_events_cache = {}
        self._current_file = None

        # The jet index identifies which jet a graph came from, which the
        # per-jet selection needs.
        kwargs["save_jet_idx"] = True
        extra_obs_to_load = list(kwargs.pop("extra_obs_to_load", None) or [])
        for branch in cut_branches(self.role, self.year, region=self.region):
            if branch not in extra_obs_to_load:
                extra_obs_to_load.append(branch)
        kwargs["extra_obs_to_load"] = extra_obs_to_load

        os.makedirs(root, exist_ok=True)
        super().__init__(root, output_dataset_name, **kwargs)

    @property
    def raw_file_names(self):
        return list(self._file_paths)

    def _processing_config(self):
        config = super()._processing_config()
        config["role"] = self.role
        config["year"] = self.year
        config["region"] = self.region
        config["cuts_fingerprint"] = cuts_fingerprint()
        config["cut_branches"] = sorted(
            cut_branches(self.role, self.year, region=self.region)
        )
        config["input_files_sha256"] = hashlib.sha256(
            "\n".join(self._file_paths).encode()
        ).hexdigest()
        return config

    def get_graphs(self, file_name):
        self._current_file = str(file_name)
        return super().get_graphs(file_name)

    def _iter_graphs_from_file(self, file_name, executor=None):
        # Remember which file is being read so get_total_initial_events can
        # look up the dataset it belongs to.
        self._current_file = str(file_name)
        yield from super()._iter_graphs_from_file(file_name, executor)

    def get_total_initial_events(self):
        """Generated events of the dataset the current file belongs to."""
        dataset = dataset_of(self._current_file, self.dataset_groups)
        if dataset not in self._initial_events_cache:
            if self.verbose:
                print(f"Summing initial events for dataset '{dataset}'")
            self._initial_events_cache[dataset] = sum_initial_events(
                self.dataset_groups[dataset]
            )
        return self._initial_events_cache[dataset]

    def process_event(
        self,
        i_start,
        i_stop,
        events,
        xsec,
        gen_events_before_selection,
        total_events,
        events_to_process,
        show_progress=None,
    ):
        """Apply the shared selections, then build graphs for what survives."""
        events_in_range = events[i_start:i_stop]
        mask = np.asarray(
            event_cut_mask(
                events_in_range, self.role, self.year, region=self.region
            ),
            dtype=bool,
        )
        if mask.shape != (len(events_in_range),):
            raise ValueError(
                "event_cut_mask must return one boolean per event, got shape "
                f"{mask.shape} for {len(events_in_range)} events"
            )

        surviving = np.flatnonzero(mask)
        if not len(surviving):
            return []

        selected_events = events_in_range[mask]
        config = self._graph_build_config()
        graphs = []

        # One event at a time, so that the jet selection can be evaluated
        # against the event a graph came from.
        for position, entry_offset in enumerate(surviving):
            event_graphs = super().process_event(
                position,
                position + 1,
                selected_events,
                xsec,
                gen_events_before_selection,
                total_events,
                events_to_process,
                show_progress=False,
            )
            for graph in event_graphs:
                jet_index = int(graph.jet_idx[0])
                if not jet_cut_mask(selected_events[position], jet_index, config):
                    continue
                if self.save_event_number:
                    graph.event_number = torch.tensor(
                        [i_start + int(entry_offset)], dtype=torch.long
                    )
                graphs.append(graph)

        return graphs
