import tempfile
import unittest
from pathlib import Path

import awkward as ak
import torch
import uproot

from data_loading.graphDataLoader import GraphDataLoader
from utils.JetGraphProducer import JetGraphProducer
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
        output["Events"] = ak.Array(events)


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


if __name__ == "__main__":
    unittest.main()
