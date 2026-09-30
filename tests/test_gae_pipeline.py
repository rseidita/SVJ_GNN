import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from architectures.ModelRegistry import build_model, training_loss, validate_model
from utils.Cuts import cuts_fingerprint, event_cut_mask, jet_cut_mask
from utils.FileLists import (
    read_file_list,
    resolve_file_list,
    shard_file_list,
    write_file_list,
)
from utils.GaeConfig import load_config
from utils.GraphBuilder import GraphBuildConfig
from utils.MetricUtils import auc, efficiency_curve, max_sic, roc_curve, sic


class TestFileLists(unittest.TestCase):
    def test_comments_and_blank_lines_are_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "files.txt")
            with open(path, "w") as list_file:
                list_file.write(
                    "# a comment\n"
                    "\n"
                    "/abs/path/a.root\n"
                    "   \n"
                    "root://cmseos.fnal.gov//store/b.root\n"
                )
            self.assertEqual(
                read_file_list(path),
                [
                    "/abs/path/a.root",
                    "root://cmseos.fnal.gov//store/b.root",
                ],
            )

    def test_empty_list_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "empty.txt")
            with open(path, "w") as list_file:
                list_file.write("# nothing here\n")
            with self.assertRaises(ValueError):
                read_file_list(path)

    def test_relative_paths_resolve_against_the_list(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "files.txt")
            with open(path, "w") as list_file:
                list_file.write(
                    "sub/a.root\n"
                    "/abs/b.root\n"
                    "root://host//store/c.root\n"
                )
            resolved = resolve_file_list(path)
            self.assertEqual(
                resolved[0], os.path.join(directory, "sub", "a.root")
            )
            self.assertEqual(resolved[1], "/abs/b.root")
            self.assertEqual(resolved[2], "root://host//store/c.root")

    def test_sharding_covers_every_file_once(self):
        paths = [f"file_{index}.root" for index in range(7)]
        shards = shard_file_list(paths, 3)
        self.assertEqual([len(shard) for shard in shards], [3, 3, 1])
        self.assertEqual([path for shard in shards for path in shard], paths)

    def test_round_trip_through_write_file_list(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "nested", "files.txt")
            paths = ["/abs/a.root", "root://host//store/b.root"]
            write_file_list(paths, path)
            self.assertEqual(read_file_list(path), paths)


class TestCuts(unittest.TestCase):
    def test_event_cut_mask_is_a_pass_through_placeholder(self):
        events = np.zeros(5)
        mask = event_cut_mask(events)
        self.assertEqual(mask.shape, (5,))
        self.assertEqual(mask.dtype, bool)
        self.assertTrue(mask.all())

    def test_jet_cut_mask_is_a_pass_through_placeholder(self):
        config = GraphBuildConfig()
        self.assertTrue(jet_cut_mask(None, 0, config))

    def test_fingerprint_is_stable(self):
        self.assertEqual(cuts_fingerprint(), cuts_fingerprint())
        self.assertEqual(len(cuts_fingerprint()), 64)


class TestModelRegistry(unittest.TestCase):
    def test_builds_both_provided_autoencoders(self):
        for spec, kwargs in (
            ("GraphAE.graphAE", {"architecture": (16, 8, 4)}),
            (
                "GraphTransformerAE.graphTransformerAE",
                {"architecture": (16, 8, 4), "heads": 2},
            ),
        ):
            model = build_model(spec, num_features=9, **kwargs)
            validate_model(model, spec)

    def test_rejects_a_malformed_spec(self):
        with self.assertRaises(ValueError):
            build_model("GraphAE", num_features=9)

    def test_rejects_an_unknown_module(self):
        with self.assertRaises(ImportError):
            build_model("NoSuchModule.model", num_features=9)

    def test_rejects_an_unknown_class(self):
        with self.assertRaises(AttributeError):
            build_model("GraphAE.noSuchClass", num_features=9)

    def test_rejects_a_model_missing_the_contract(self):
        class Incomplete:
            pass

        with self.assertRaises(TypeError):
            validate_model(Incomplete())

    def test_a_model_defining_training_loss_overrides_the_fallback(self):
        class CustomLoss:
            def forward(self, x, edge_index):
                return x

            def score_batch(self, x, edge_index, batch, alpha=1.0):
                return x, x, x

            def training_loss(self, batch, alpha=1.0):
                return "total", "edges", "features"

        model = validate_model(CustomLoss())
        self.assertEqual(
            training_loss(model, batch=None), ("total", "edges", "features")
        )


class TestMetrics(unittest.TestCase):
    def test_efficiency_falls_from_one_to_zero(self):
        thresholds, efficiency = efficiency_curve([1.0, 2.0, 3.0, 4.0])
        self.assertAlmostEqual(efficiency[0], 1.0)
        self.assertAlmostEqual(efficiency[-1], 0.0)
        self.assertTrue(np.all(np.diff(efficiency) <= 0))
        self.assertEqual(len(thresholds), len(efficiency))

    def test_weights_are_respected(self):
        _, efficiency = efficiency_curve([0.0, 1.0], weights=[3.0, 1.0])
        # Cutting just above the first entry keeps only a quarter of the weight.
        self.assertAlmostEqual(efficiency[1], 0.25)

    def test_separated_distributions_give_a_high_auc(self):
        background = np.linspace(0.0, 1.0, 200)
        signal = np.linspace(2.0, 3.0, 200)
        fpr, tpr, _ = roc_curve(signal, background)
        self.assertGreater(auc(fpr, tpr), 0.99)
        self.assertGreater(max_sic(tpr, fpr, min_fpr=1e-3), 1.0)

    def test_identical_distributions_give_a_diagonal_roc(self):
        values = np.linspace(0.0, 1.0, 200)
        fpr, tpr, _ = roc_curve(values, values)
        np.testing.assert_allclose(fpr, tpr)
        self.assertAlmostEqual(auc(fpr, tpr), 0.5, places=2)

    def test_sic_drops_points_below_the_floor(self):
        fpr = np.array([0.0, 1e-8, 0.5, 1.0])
        tpr = np.array([0.0, 0.1, 0.8, 1.0])
        kept, values = sic(tpr, fpr, min_fpr=1e-3)
        self.assertEqual(len(kept), 2)
        self.assertTrue(np.all(np.isfinite(values)))


class TestConfig(unittest.TestCase):
    def setUp(self):
        self.repository = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )

    def test_shipped_configuration_loads(self):
        config = load_config(
            os.path.join(self.repository, "gae_config.json")
        )
        self.assertIn("smoke", config.samples)
        roles = {sample.role for sample in config.samples.values()}
        self.assertEqual(roles, {"signal", "background", "data"})

    def test_graph_build_config_drops_producer_only_settings(self):
        config = load_config(
            os.path.join(self.repository, "gae_config.json")
        )
        build_config = config.graph_build_config(label=1.0)
        self.assertTrue(build_config.use_lund_decomp)
        self.assertEqual(build_config.n_lund_vars, 5)
        self.assertEqual(build_config.label, 1.0)
        self.assertFalse(hasattr(build_config, "n_store_jets"))

    def test_unknown_role_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "bad.json")
            with open(path, "w") as config_file:
                json.dump(
                    {
                        "samples": {
                            "x": {"role": "sideband", "file_list": "x.txt"}
                        }
                    },
                    config_file,
                )
            with self.assertRaises(ValueError):
                load_config(path)

    def test_unknown_sample_is_rejected(self):
        config = load_config(
            os.path.join(self.repository, "gae_config.json")
        )
        with self.assertRaises(KeyError):
            config.sample("not_a_sample")


class TestFileListGraphProducer(unittest.TestCase):
    def test_raw_file_names_come_from_the_file_list(self):
        from utils.JetGraphProducer import FileListGraphProducer

        paths = ["/abs/a.root", "root://host//store/b.root"]
        producer = FileListGraphProducer.__new__(FileListGraphProducer)
        producer._file_paths = list(paths)
        self.assertEqual(producer.raw_file_names, paths)

    def test_requires_exactly_one_input_source(self):
        from utils.JetGraphProducer import FileListGraphProducer

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                FileListGraphProducer(directory, "dataset")
            with self.assertRaises(ValueError):
                FileListGraphProducer(
                    directory,
                    "dataset",
                    file_list="a.txt",
                    file_paths=["b.root"],
                )


if __name__ == "__main__":
    unittest.main()
