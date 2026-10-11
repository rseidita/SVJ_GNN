import tempfile
import unittest
from pathlib import Path

import awkward as ak
import torch
import uproot

from architectures.GraphAE import graphAE
from data_loading.graphDataLoader import (
    ChunkAwareWeightedSampler,
    GraphDataLoader,
)
from utils.GraphBuilder import GraphBuildConfig, build_pfnano_jet_graph
from utils.JetGraphProducer import JetGraphProducer
from utils.RootGraphScorer import RootGraphScorer
from utils.ScoreProducer import ScoreProducer


def write_fixture(root):
    events = []
    for event_index in range(4):
        events.append({
            "nFatJet": 2,
            "FatJet_pt": [200.0 + event_index, 100.0 + event_index],
            "FatJet_eta": [0.1, -0.2],
            "FatJet_phi": [3.13, -0.4],
            "FatJetPFCands_jetIdx": [0, 0, 0, 1, 1, 1],
            "FatJetPFCands_pFCandsIdx": [0, 1, 2, 3, 4, 5],
            "PFCands_pt": [30.0, 20.0, 10.0, 25.0, 15.0, 5.0],
            "PFCands_eta": [0.11, 0.08, -0.1, -0.19, -0.25, 0.2],
            "PFCands_phi": [3.12, -3.13, 2.9, -0.41, -0.35, 0.1],
            "PFCands_pdgId": [211, 22, 13, 211, 11, 22],
            "PFCands_mass": [0.14, 0.0, 0.105, 0.14, 0.0005, 0.0],
        })
    with uproot.recreate(root / "fixture.root") as output:
        output["Events"] = {
            key: ak.Array([e[key] for e in events]) for key in events[0]
        }


class ProductionTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory(
            prefix="svj_production_test_"
        )
        self.root = Path(self.temporary_directory.name)
        write_fixture(self.root)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def producer(self, name, **kwargs):
        options = dict(
            root=str(self.root),
            output_dataset_name=name,
            n_store_jets=2,
            use_lund_decomp=True,
            n_lund_vars=5,
            save_energy_fractions=True,
            save_4vectors_in_lund_tree=True,
            weights=None,
            save_event_number=True,
            save_jet_idx=True,
            graphs_per_chunk=3,
            event_batch_size=1,
            verbose=False,
        )
        options.update(kwargs)
        return JetGraphProducer(**options)

    def test_parallel_matches_sequential(self):
        sequential = self.producer("sequential", num_workers=1)
        parallel = self.producer(
            "parallel", num_workers=2, worker_prefetch=2
        )
        self.assertEqual(len(sequential), 8)
        self.assertEqual(len(parallel), 8)
        for left, right in zip(sequential, parallel):
            self.assertTrue(torch.equal(left.edge_index, right.edge_index))
            self.assertTrue(torch.allclose(left.x, right.x))
            self.assertTrue(
                torch.equal(left.event_number, right.event_number)
            )
            self.assertTrue(torch.equal(left.jet_idx, right.jet_idx))

    def test_cache_configuration_mismatch_is_rejected(self):
        self.producer("cache")
        with self.assertRaisesRegex(RuntimeError, "use_lund_decomp"):
            self.producer(
                "cache",
                use_lund_decomp=False,
                save_4vectors_in_lund_tree=False,
            )

    def test_legacy_cache_without_configuration_is_rejected(self):
        processed = self.root / "processed"
        processed.mkdir(exist_ok=True)
        torch.save(
            {
                "n_graphs": 0,
                "n_chunks": 0,
                "chunk_size": 3,
                "w": torch.empty(0),
                "jet_idx": None,
            },
            processed / "legacy_meta.pt",
        )
        with self.assertRaisesRegex(
            RuntimeError, "without producer configuration metadata"
        ):
            self.producer("legacy")

    def test_chunk_names_and_deprecated_aliases(self):
        dataset = self.producer(
            "aliases",
            graphs_per_chunk=None,
            event_batch_size=None,
            chunk_size=4,
            processing_batch_size=2,
        )
        self.assertEqual(dataset.graphs_per_chunk, 4)
        self.assertEqual(dataset.chunk_size, 4)
        self.assertEqual(dataset.event_batch_size, 2)
        self.assertEqual(dataset.processing_batch_size, 2)
        self.assertTrue(
            (self.root / "processed" / "aliases_chunk_1.pt").exists()
        )

    def test_loader_filters_weights_after_jet_filtering(self):
        self.producer("loader")
        loader = GraphDataLoader(
            [[str(self.root), "loader"]],
            batch_size=2,
            jet_idx=1,
            sample=True,
        )
        self.assertEqual(len(loader.dataset), 4)
        self.assertEqual(len(loader.sampler.weights), 4)
        batch = next(iter(loader))
        self.assertEqual(batch.num_graphs, 2)
        self.assertTrue(torch.equal(batch.jet_idx, torch.ones(2, dtype=torch.int)))

    @staticmethod
    def loader_identities(loader):
        identities = []
        for batch in loader:
            identities.extend(zip(
                batch.event_number.view(-1).tolist(),
                batch.jet_idx.view(-1).tolist(),
            ))
        return identities

    def test_reproducible_randomized_split_state(self):
        self.producer("splits")
        paths = [[str(self.root), "splits"]]
        train = GraphDataLoader(
            paths,
            split_windows=[[0.0, 0.5]],
            split_seed=1234,
            sampling="sequential",
            batch_size=2,
        )
        state = train.get_split_state()
        validation = GraphDataLoader(
            paths,
            split_windows=[[0.5, 1.0]],
            split_state=state,
            sampling="sequential",
            batch_size=2,
        )
        repeated_train = GraphDataLoader(
            paths,
            split_windows=[[0.0, 0.5]],
            split_seed=1234,
            sampling="sequential",
            batch_size=2,
        )

        train_ids = self.loader_identities(train)
        validation_ids = self.loader_identities(validation)
        self.assertEqual(
            train_ids, self.loader_identities(repeated_train)
        )
        self.assertTrue(set(train_ids).isdisjoint(validation_ids))
        self.assertEqual(len(set(train_ids) | set(validation_ids)), 8)

    def test_split_membership_is_traversed_in_storage_order(self):
        self.producer("storage_order")
        loader = GraphDataLoader(
            [[str(self.root), "storage_order"]],
            split_windows=[[0.0, 0.5]],
            split_seed=9876,
            sampling="sequential",
            batch_size=2,
        )
        record = next(iter(loader.get_split_state()["datasets"].values()))
        expected_membership = set(record["permutation"][:4].tolist())

        dataset_view = loader.dataset.datasets[0]
        selected_indices = torch.as_tensor(
            dataset_view.indices(), dtype=torch.long
        )
        self.assertEqual(set(selected_indices.tolist()), expected_membership)
        self.assertTrue(
            torch.equal(selected_indices, selected_indices.sort().values)
        )

    def test_epoch_shuffle_is_independent_of_fixed_split(self):
        self.producer("epoch_shuffle")
        loader = GraphDataLoader(
            [[str(self.root), "epoch_shuffle"]],
            split_seed=4321,
            sampling="shuffle",
            generator=torch.Generator().manual_seed(99),
            batch_size=2,
        )
        first_epoch = self.loader_identities(loader)
        second_epoch = self.loader_identities(loader)
        self.assertEqual(set(first_epoch), set(second_epoch))
        self.assertNotEqual(first_epoch, second_epoch)

    def test_stale_split_state_is_rejected(self):
        self.producer("stale")
        paths = [[str(self.root), "stale"]]
        loader = GraphDataLoader(paths, split_seed=1)
        state = loader.get_split_state()

        metadata_file = self.root / "processed" / "stale_meta.pt"
        metadata = torch.load(metadata_file, weights_only=False)
        metadata["test_mutation"] = True
        torch.save(metadata, metadata_file)

        with self.assertRaisesRegex(
            ValueError, "Processed metadata changed"
        ):
            GraphDataLoader(paths, split_state=state)

    def test_loader_argument_validation(self):
        self.producer("validation")
        paths = [[str(self.root), "validation"]]
        with self.assertRaisesRegex(ValueError, "Invalid split window"):
            GraphDataLoader(paths, split_windows=[[0.8, 0.2]])
        with self.assertRaisesRegex(
            ValueError, "only meaningful with sampling='weighted'"
        ):
            GraphDataLoader(
                paths,
                sampling="sequential",
                dataset_fraction=[1.0],
            )
        with self.assertRaisesRegex(ValueError, "sampling must be one of"):
            GraphDataLoader(paths, sampling="randomish")

    def test_dataset_fractions_control_sampling_mass(self):
        self.producer("fraction_a")
        self.producer("fraction_b")
        loader = GraphDataLoader(
            [
                [str(self.root), "fraction_a"],
                [str(self.root), "fraction_b"],
            ],
            sampling="weighted",
            sampling_weights="uniform",
            dataset_fraction=[1.0, 3.0],
            batch_size=2,
        )
        weights = loader.sampler.weights
        self.assertAlmostEqual(weights[:8].sum().item(), 0.25)
        self.assertAlmostEqual(weights[8:].sum().item(), 0.75)

    def test_chunk_aware_sampler_preserves_weighted_draw(self):
        weights = torch.arange(1, 13, dtype=torch.double)
        chunk_ids = torch.arange(12) // 3
        seed = 2468
        expected = torch.multinomial(
            weights,
            200,
            replacement=True,
            generator=torch.Generator().manual_seed(seed),
        )
        sampler = ChunkAwareWeightedSampler(
            weights,
            chunk_ids,
            num_samples=200,
            generator=torch.Generator().manual_seed(seed),
        )
        actual = torch.tensor(list(sampler))

        self.assertTrue(
            torch.equal(actual.sort().values, expected.sort().values)
        )
        traversed_chunks = chunk_ids[actual]
        transitions = (traversed_chunks[1:] != traversed_chunks[:-1]).sum()
        self.assertLessEqual(transitions.item(), 3)

    def test_chunk_aware_weighted_loader_changes_epoch_draw(self):
        self.producer("chunk_weighted")
        loader = GraphDataLoader(
            [[str(self.root), "chunk_weighted"]],
            sampling="weighted",
            generator=torch.Generator().manual_seed(1357),
            batch_size=2,
        )
        first_epoch = self.loader_identities(loader)
        second_epoch = self.loader_identities(loader)
        self.assertEqual(len(first_epoch), len(loader.dataset))
        self.assertEqual(len(second_epoch), len(loader.dataset))
        self.assertNotEqual(first_epoch, second_epoch)

    def test_treemaker_is_explicitly_unsupported(self):
        with self.assertRaisesRegex(
            ValueError, "TreeMaker support has been removed"
        ):
            self.producer("obsolete", input_format="TreeMaker2")
        with self.assertRaisesRegex(
            ValueError, "TreeMaker support has been removed"
        ):
            ScoreProducer(
                str(self.root / "fixture.root"),
                input_format="TreeMaker2",
                model=None,
            )

    def test_streaming_scorer_preserves_order_without_pt_files(self):
        config = GraphBuildConfig(
            use_lund_decomp=True,
            n_lund_vars=5,
            save_energy_fractions=True,
            save_4vectors_in_lund_tree=True,
        )
        model = graphAE(num_features=12, architecture=(8, 4))
        file_name = self.root / "fixture.root"

        single_graph = RootGraphScorer(
            model,
            config,
            device="cpu",
            root_entry_batch_size=1,
            graph_batch_size=1,
        )
        batched = RootGraphScorer(
            model,
            config,
            device="cpu",
            root_entry_batch_size=3,
            graph_batch_size=5,
            num_workers=2,
            worker_prefetch=1,
        )
        reference = single_graph.score_file(file_name)
        result = batched.score_file(file_name)

        self.assertEqual(ak.num(result, axis=0), 4)
        self.assertEqual(ak.to_list(ak.num(result, axis=1)), [2, 2, 2, 2])
        self.assertTrue(
            torch.allclose(
                torch.tensor(ak.flatten(reference).to_list()),
                torch.tensor(ak.flatten(result).to_list()),
                rtol=1e-5,
                atol=1e-6,
            )
        )

        with uproot.open(f"{file_name}:Events") as tree:
            event = tree.arrays(library="ak", entry_stop=1)[0]
        graph, _ = build_pfnano_jet_graph(event, 0, config)
        model.eval()
        with torch.inference_mode():
            legacy_score = (
                model.edges_loss(
                    graph.x,
                    graph.edge_index,
                    torch.empty((2, 0), dtype=torch.long),
                )
                + model.features_loss(graph.x, graph.edge_index)
            )
            event_scores = model.score_event(graph)
            tensor_scores = model.score_event(
                graph.x, graph.edge_index
            )
        self.assertTrue(
            torch.isclose(
                torch.tensor(result[0, 0], dtype=legacy_score.dtype),
                legacy_score,
                rtol=1e-5,
                atol=1e-6,
            )
        )
        self.assertTrue(torch.isclose(event_scores[0], legacy_score))
        for graph_score, tensor_score in zip(
            event_scores, tensor_scores
        ):
            self.assertTrue(torch.equal(graph_score, tensor_score))
        self.assertEqual(list(self.root.rglob("*.pt")), [])

    def test_streaming_scorer_yields_contiguous_entry_chunks(self):
        config = GraphBuildConfig(
            use_lund_decomp=True,
            n_lund_vars=3,
            save_energy_fractions=False,
        )
        scorer = RootGraphScorer(
            graphAE(num_features=3, architecture=(4, 2)),
            config,
            device="cpu",
            root_entry_batch_size=3,
            graph_batch_size=4,
        )
        chunks = list(scorer.iter_file(self.root / "fixture.root"))
        self.assertEqual(
            [(chunk.entry_start, chunk.entry_stop) for chunk in chunks],
            [(0, 3), (3, 4)],
        )

    def test_shard_pre_transform_and_pre_filter_files(self):
        # Produce shard 0
        self.producer("test_sample_shard0")
        processed_dir = self.root / "processed"
        self.assertTrue((processed_dir / "test_sample_shard0_pre_transform.pt").exists())
        self.assertTrue((processed_dir / "test_sample_shard0_pre_filter.pt").exists())
        self.assertTrue((processed_dir / "pre_transform.pt").exists())
        self.assertTrue((processed_dir / "pre_filter.pt").exists())

        # Produce shard 1 in the same directory without conflicts
        self.producer("test_sample_shard1")
        self.assertTrue((processed_dir / "test_sample_shard1_pre_transform.pt").exists())
        self.assertTrue((processed_dir / "test_sample_shard1_pre_filter.pt").exists())

        # Reloading shard 0 shouldn't trigger warnings
        reloaded = self.producer("test_sample_shard0")
        self.assertEqual(len(reloaded), 8)

        # Reloading shard 0 with different pre_transform should warn
        with self.assertWarnsRegex(UserWarning, "pre_transform.*differs"):
            self.producer("test_sample_shard0", pre_transform=lambda g: g, validate_cache_config=False)


if __name__ == "__main__":
    unittest.main()
