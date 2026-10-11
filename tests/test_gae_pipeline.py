import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import awkward as ak
import numpy as np

from architectures.ModelRegistry import build_model, training_loss, validate_model
from utils.Cuts import (
    HEM_FIRST_RUN,
    POST_HEM_FRACTION,
    add_cutflow,
    cut_branches,
    cutflow,
    cuts_fingerprint,
    event_cut_mask,
    format_cutflow,
    jet_cut_mask,
    post_hem_mask,
)
from utils.GaeConfig import load_config
from utils.GraphBuilder import GraphBuildConfig
from utils.InputJson import (
    dataset_groups,
    dataset_of,
    flat_files,
    load_input_json,
    normalize_input_path,
    resolve_file_selection,
    score_output_basename,
    shard_files,
    write_input_json,
)
from utils.MetricUtils import auc, efficiency_curve, max_sic, roc_curve, sic


class TestInputJson(unittest.TestCase):
    def setUp(self):
        self.datasets = {
            "QCD_HT500to700": ["root://host//store/b.root", "root://host//store/a.root"],
            "QCD_HT300to500": ["root://host//store/c.root"],
        }

    def test_round_trip_sorts_datasets_and_files(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "nested", "2018_background.json")
            write_input_json(path, 2018, "background", "root://host//base/", self.datasets)
            spec = load_input_json(path)
        self.assertEqual(spec["year"], "2018")
        self.assertEqual(spec["type"], "background")
        self.assertEqual(list(spec["datasets"]), ["QCD_HT300to500", "QCD_HT500to700"])
        self.assertEqual(
            spec["datasets"]["QCD_HT500to700"],
            ["root://host//store/a.root", "root://host//store/b.root"],
        )

    def test_empty_json_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "empty.json")
            with open(path, "w") as json_file:
                json.dump({"datasets": {"x": []}}, json_file)
            with self.assertRaises(ValueError):
                load_input_json(path)

    def test_groups_can_be_restricted_to_named_datasets(self):
        spec = {"datasets": self.datasets}
        groups = dataset_groups(spec, ["QCD_HT300to500"])
        self.assertEqual(list(groups), ["QCD_HT300to500"])
        with self.assertRaises(KeyError):
            dataset_groups(spec, ["TTJets"])

    def test_flat_files_and_dataset_lookup_are_consistent(self):
        files = flat_files(self.datasets)
        self.assertEqual(len(files), 3)
        self.assertEqual(files[0], "root://host//store/c.root")
        for file_path in files:
            self.assertIn(file_path, self.datasets[dataset_of(file_path, self.datasets)])
        with self.assertRaises(KeyError):
            dataset_of("nope.root", self.datasets)

    def test_sharding_covers_every_file_once(self):
        paths = [f"file_{index}.root" for index in range(7)]
        shards = shard_files(paths, 3)
        self.assertEqual([len(shard) for shard in shards], [3, 3, 1])
        self.assertEqual([path for shard in shards for path in shard], paths)

    def test_normalize_input_path_repairs_normpath_damage_to_xrootd(self):
        url = (
            "root://cmsdcache-kit-disk.gridka.de:1094//store/user/foo/"
            "part-0.root"
        )
        broken = os.path.normpath(url)
        self.assertTrue(broken.startswith("root:/"))
        self.assertEqual(
            normalize_input_path(broken),
            "root://cmsdcache-kit-disk.gridka.de:1094/store/user/foo/part-0.root",
        )

    def test_score_output_basename_uses_role_and_part_index(self):
        remote = (
            "root://host//store/part-7.root"
        )
        self.assertEqual(
            score_output_basename("data", remote), "data_part-7.root"
        )
        self.assertEqual(
            score_output_basename("background", remote), "qcd_part-7.root"
        )
        self.assertEqual(
            score_output_basename(
                "signal",
                remote,
                sample_name="signal_2018_s-channel_mMed-1000_mDark-20_rinv-0.3",
            ),
            "signal_mMed-1000_mDark-20_rinv-0.3_part-7.root",
        )

    def test_score_output_basename_disambiguates_multi_dataset_and_years(self):
        remote = "root://host//store/part-7.root"
        data_2018a = score_output_basename("data", remote, sample_name="data_2018", dataset="Run2018A")
        data_2018b = score_output_basename("data", remote, sample_name="data_2018", dataset="Run2018B")
        data_2017b = score_output_basename("data", remote, sample_name="data_2017", dataset="Run2017B")
        self.assertEqual(data_2018a, "data_2018_Run2018A_part-7.root")
        self.assertEqual(data_2018b, "data_2018_Run2018B_part-7.root")
        self.assertEqual(data_2017b, "data_2017_Run2017B_part-7.root")
        self.assertEqual(len({data_2018a, data_2018b, data_2017b}), 3)

        qcd_2018 = score_output_basename("background", remote, sample_name="background_2018", dataset="QCD_HT500to700")
        qcd_2017 = score_output_basename("background", remote, sample_name="background_2017", dataset="QCD_HT500to700")
        self.assertEqual(qcd_2018, "qcd_2018_QCD_HT500to700_part-7.root")
        self.assertEqual(qcd_2017, "qcd_2017_QCD_HT500to700_part-7.root")
        self.assertNotEqual(qcd_2018, qcd_2017)

        sig_2018 = score_output_basename(
            "signal", remote,
            sample_name="signal_2018_s-channel_mMed-1000_mDark-20_rinv-0.3",
            dataset="s-channel_mMed-1000_mDark-20_rinv-0.3"
        )
        sig_2017 = score_output_basename(
            "signal", remote,
            sample_name="signal_2017_s-channel_mMed-1000_mDark-20_rinv-0.3",
            dataset="s-channel_mMed-1000_mDark-20_rinv-0.3"
        )
        self.assertEqual(sig_2018, "signal_2018_mMed-1000_mDark-20_rinv-0.3_part-7.root")
        self.assertEqual(sig_2017, "signal_2017_mMed-1000_mDark-20_rinv-0.3_part-7.root")
        self.assertNotEqual(sig_2018, sig_2017)


    def test_resolve_file_selection_normalizes_xrootd_paths(self):
        paths = [
            "root://host/store/a.root",
            "root://host/store/b.root",
        ]
        selected = resolve_file_selection(
            paths, [os.path.normpath(paths[0])]
        )
        self.assertEqual(selected, [paths[0]])


def fake_events(**overrides):
    """Five events exercising each selection; defaults all pass as MC 2017."""
    columns = {
        "run": [1, 1, 1, 1, 1],
        "lumSec": [1, 2, 3, 4, 5],
        "event": [10, 20, 30, 40, 50],
        "nFatJet": [2, 2, 2, 2, 2],
        "DeltaEtaJ0J1FatJet": [1.8, -1.9, 2.0, 1.6, 2.1],
        "FatJet_isGood": [[True, True]] * 5,
        "FatJet_pt": [[500.0, 300.0]] * 5,
        "FatJet_photonEnergyFraction": [[0.1, 0.1]] * 5,
        "HEMVeto": [True] * 5,
    }
    columns.update(overrides)
    return ak.Array(columns)


class TestCuts(unittest.TestCase):
    def test_cut_branches_only_need_hem_veto_for_2018_mc(self):
        self.assertIn("HEMVeto", cut_branches("background", "2018"))
        self.assertIn("HEMVeto", cut_branches("signal", 2018))
        self.assertNotIn("HEMVeto", cut_branches("data", "2018"))
        self.assertNotIn("HEMVeto", cut_branches("background", "2017"))

    def test_everything_passes_by_default(self):
        mask = event_cut_mask(fake_events(), "background", "2017")
        self.assertEqual(mask.shape, (5,))
        self.assertEqual(mask.dtype, bool)
        self.assertTrue(mask.all())

    def test_leading_two_jets_must_be_good(self):
        events = fake_events(
            FatJet_isGood=[
                [True, True],
                [False, True],
                [True, False],
                [True, True, False],
                [True],
            ],
            nFatJet=[2, 2, 2, 3, 1],
        )
        mask = event_cut_mask(events, "signal", "2017")
        self.assertEqual(mask.tolist(), [True, False, False, True, False])

    def test_delta_eta_control_region_applies_to_every_role(self):
        events = fake_events(
            run=[300000] * 5,
            DeltaEtaJ0J1FatJet=[1.4, 1.5, 1.8, -2.1, 2.3],
        )
        expected = [False, False, True, True, False]
        for role in ("data", "background", "signal"):
            self.assertEqual(
                event_cut_mask(events, role, "2017").tolist(), expected, role
            )

    def test_cutflow_counts_are_cumulative_and_match_the_mask(self):
        events = fake_events(
            FatJet_isGood=[[True, True], [False, True]] + [[True, True]] * 3,
            DeltaEtaJ0J1FatJet=[1.8, 1.8, 2.5, 1.8, 1.8],
        )
        counts = cutflow(events, "background", "2017")
        self.assertEqual(list(counts.values()), [5, 4, 3])
        self.assertEqual(
            list(counts.values())[-1],
            int(event_cut_mask(events, "background", "2017").sum()),
        )
        self.assertEqual(len(cutflow(events, "data", "2018")), 4)
        self.assertEqual(len(cutflow(events, "signal", "2018")), 5)
        total = add_cutflow({}, counts)
        total = add_cutflow(total, counts)
        self.assertEqual(list(total.values()), [10, 8, 6])
        table = format_cutflow("x", total)
        self.assertIn("skim", table)
        self.assertIn("0.6000", table)

    def test_gap_jet_veto_applies_to_post_hem_data_runs(self):
        events = fake_events(
            run=[HEM_FIRST_RUN - 1, HEM_FIRST_RUN, HEM_FIRST_RUN + 5, 320000, 320000],
            FatJet_photonEnergyFraction=[[0.7, 0.1]] * 4 + [[0.1, 0.7]],
            FatJet_pt=[[500.0, 300.0]] * 3 + [[350.0, 300.0], [500.0, 300.0]],
        )
        mask = event_cut_mask(events, "data", "2018")
        # pre-HEM run passes; post-HEM with pt > 400 and fraction > 0.6 fail;
        # low pt or low leading fraction pass.
        self.assertEqual(mask.tolist(), [True, False, False, True, True])
        # Not a 2018 sample: nothing vetoed by the gap jet criterion.
        self.assertTrue(event_cut_mask(events, "data", "2017").all())

    def test_mc_2018_split_is_deterministic_and_close_to_the_fraction(self):
        n = 20000
        events = ak.Array(
            {
                "run": np.ones(n, dtype=np.int64),
                "lumSec": np.arange(n) // 100,
                "event": np.arange(n),
            }
        )
        first = post_hem_mask(events, "background", "2018")
        second = post_hem_mask(events, "background", "2018")
        np.testing.assert_array_equal(first, second)
        self.assertAlmostEqual(first.mean(), POST_HEM_FRACTION, delta=0.02)
        self.assertFalse(post_hem_mask(events, "background", "2017").any())

    def test_hem_veto_branch_only_matters_for_the_post_hem_mc_fraction(self):
        n = 2000
        base = {
            "run": np.ones(n, dtype=np.int64),
            "lumSec": np.ones(n, dtype=np.int64),
            "event": np.arange(n),
            "nFatJet": np.full(n, 2),
            "DeltaEtaJ0J1FatJet": np.full(n, 1.8),
            "FatJet_isGood": [[True, True]] * n,
            "FatJet_pt": [[500.0, 300.0]] * n,
            "FatJet_photonEnergyFraction": [[0.1, 0.1]] * n,
        }
        vetoed = ak.Array({**base, "HEMVeto": np.zeros(n, dtype=bool)})
        passing = ak.Array({**base, "HEMVeto": np.ones(n, dtype=bool)})
        post_hem = post_hem_mask(vetoed, "background", "2018")
        mask = event_cut_mask(vetoed, "background", "2018")
        np.testing.assert_array_equal(mask, ~post_hem)
        self.assertTrue(event_cut_mask(passing, "background", "2018").all())

    def test_jet_cut_mask_is_a_pass_through(self):
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
        for sample in config.samples.values():
            self.assertIn(sample.year, ("2017", "2018"))
            self.assertTrue(sample.input_json_path.endswith(".json"))
        smoke = config.sample("smoke")
        self.assertEqual(smoke.year, "2018")
        self.assertEqual(list(smoke.groups()), ["Run2018A_part14"])

    def test_per_dataset_signal_is_expanded_into_one_sample_per_point(self):
        config = load_config(
            os.path.join(self.repository, "gae_config.json")
        )
        self.assertNotIn("signal_2018", config.samples)
        points = config.select(["signal_2018"])
        self.assertGreater(len(points), 20)
        for point in points:
            self.assertEqual(point.group, "signal_2018")
            self.assertEqual(point.role, "signal")
            self.assertEqual(len(point.datasets), 1)
            self.assertTrue(point.name.startswith("signal_2018_s-channel"))
            self.assertEqual(list(point.groups()), point.datasets)
        # Exact names and groups mix, and unknown names fail.
        selected = config.select(["smoke", "signal_2018"])
        self.assertEqual(selected[0].name, "smoke")
        self.assertEqual(len(selected), 1 + len(points))
        with self.assertRaises(KeyError):
            config.select(["nope"])
        # QCD-only background sample keeps just the six HT bins.
        self.assertEqual(len(config.sample("qcd_2018").groups()), 6)

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
                            "x": {
                                "role": "sideband",
                                "year": "2018",
                                "input_json": "x.json",
                            }
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
    def test_raw_file_names_come_from_the_given_paths(self):
        from utils.JetGraphProducer import FileListGraphProducer

        paths = ["/abs/a.root", "root://host//store/b.root"]
        producer = FileListGraphProducer.__new__(FileListGraphProducer)
        producer._file_paths = list(paths)
        self.assertEqual(producer.raw_file_names, paths)

    def test_initial_events_are_summed_per_dataset_and_cached(self):
        from utils.JetGraphProducer import FileListGraphProducer

        groups = {"A": ["a0.root", "a1.root"], "B": ["b0.root"]}
        producer = FileListGraphProducer.__new__(FileListGraphProducer)
        producer.dataset_groups = groups
        producer.verbose = False
        producer._initial_events_cache = {"A": 7, "B": 3}
        producer._current_file = "a1.root"
        self.assertEqual(producer.get_total_initial_events(), 7)
        producer._current_file = "b0.root"
        self.assertEqual(producer.get_total_initial_events(), 3)

    def test_requires_input_files(self):
        from utils.JetGraphProducer import FileListGraphProducer

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                FileListGraphProducer(
                    directory, "dataset", file_paths=[], role="data", year="2018"
                )


class TestMultiYearAndDisambiguatedScoring(unittest.TestCase):
    def test_multi_year_scoring_and_plotting_flow(self):
        import numpy as np
        import uproot
        from plot_gae import load_samples_from_config, combined_overlays, score_paths_for_sample
        from utils.InputJson import score_output_basename
        from utils.GaeConfig import load_config

        repository = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config = load_config(os.path.join(repository, "gae_config.json"))

        with tempfile.TemporaryDirectory() as scores_dir:
            files_to_create = [
                ("data", "data_2018", "Run2018A", "part-0.root", [1.0, 2.0], [1.0, 1.0]),
                ("data", "data_2018", "Run2018B", "part-0.root", [3.0], [1.0]),
                ("data", "data_2017", "Run2017B", "part-0.root", [4.0, 5.0], [1.0, 1.0]),
                ("background", "background_2018", "QCD_HT500to700", "part-0.root", [10.0], [0.5]),
                ("background", "background_2017", "QCD_HT500to700", "part-0.root", [20.0], [0.8]),
                ("signal", "signal_2018_s-channel_mMed-1000_mDark-20_rinv-0.3", "s-channel_mMed-1000_mDark-20_rinv-0.3", "part-0.root", [100.0], [1.0]),
                ("signal", "signal_2017_s-channel_mMed-1000_mDark-20_rinv-0.3", "s-channel_mMed-1000_mDark-20_rinv-0.3", "part-0.root", [200.0], [1.0]),
            ]
            created_files = set()
            for role, sname, dset, fname, sc, wt in files_to_create:
                out_name = score_output_basename(role, fname, sample_name=sname, dataset=dset)
                self.assertNotIn(out_name, created_files, f"Collision detected for {out_name}!")
                created_files.add(out_name)
                full_p = os.path.join(scores_dir, out_name)
                with uproot.recreate(full_p) as out_f:
                    out_f["Scores"] = {
                        "score": np.asarray(sc, dtype=np.float32),
                        "weight": np.asarray(wt, dtype=np.float32),
                    }

            paths_2018 = score_paths_for_sample(scores_dir, config.sample("data_2018"))
            self.assertEqual(len(paths_2018), 2)

            paths_2017 = score_paths_for_sample(scores_dir, config.sample("data_2017"))
            self.assertEqual(len(paths_2017), 1)

            loaded = load_samples_from_config(config, scores_dir, ["data_2018", "data_2017"])
            self.assertEqual(len(loaded["data_2018"][1]), 3)
            self.assertEqual(len(loaded["data_2017"][1]), 2)

            overlays, roles = combined_overlays(scores_dir)
            self.assertIn("data", overlays)
            self.assertEqual(len(overlays["data"][0]), 5)
            self.assertIn("qcd", overlays)
            self.assertEqual(len(overlays["qcd"][0]), 2)


if __name__ == "__main__":
    unittest.main()
