"""test_plotting_framework.py - Unit and integration tests for the plotting framework."""

from __future__ import annotations
import os
import shutil
import tempfile
import unittest

import awkward as ak
import numpy as np

from plotting.hist_reader import (
    TH1Accumulator,
    bkg_family_name,
    bkg_family_groups,
    compute_chunk_weights,
    load_histograms_from_root,
    save_histograms_to_root,
)
from plotting.plot_cuts import (
    cut_branches,
    get_event_selection,
    post_hem_mask,
)
from plotting.plot_stack import plot_stacked_variable
from plotting.plot_variables import (
    VARIABLES,
    extract_variable,
    required_branches_for_vars,
)


class TestPlotVariables(unittest.TestCase):
    def test_variable_definitions(self):
        self.assertIn("MT01FatJetMET", VARIABLES)
        self.assertIn("RTFatJet", VARIABLES)
        self.assertIn("FatJet1_pt", VARIABLES)
        for var, cfg in VARIABLES.items():
            self.assertIn("branch", cfg, f"Missing 'branch' in {var}")
            self.assertIn("bins", cfg, f"Missing 'bins' in {var}")
            self.assertIn("xlabel", cfg, f"Missing 'xlabel' in {var}")
            self.assertGreater(len(cfg["bins"]), 1, f"Bins too short in {var}")

    def test_variable_extraction(self):
        events = ak.Array({
            "MT01FatJetMET": [700.0, 1200.0, 1500.0],
            "FatJet_pt": [[500.0, 300.0], [600.0], []],
        })
        mt = extract_variable(events, "MT01FatJetMET")
        np.testing.assert_allclose(mt, [700.0, 1200.0, 1500.0])

        fj1_pt = extract_variable(events, "FatJet1_pt")
        np.testing.assert_allclose(fj1_pt[:2], [500.0, 600.0])
        self.assertTrue(np.isnan(fj1_pt[2]))

        fj2_pt = extract_variable(events, "FatJet2_pt")
        self.assertEqual(fj2_pt[0], 300.0)
        self.assertTrue(np.isnan(fj2_pt[1]))
        self.assertTrue(np.isnan(fj2_pt[2]))


class TestPlotCuts(unittest.TestCase):
    def test_cut_branches(self):
        mc_branches = cut_branches("background", "2018")
        self.assertIn("HEMVeto", mc_branches)
        self.assertIn("DeltaEtaJ0J1FatJet", mc_branches)

        data_branches = cut_branches("data", "2018")
        self.assertNotIn("HEMVeto", data_branches)

    def test_event_selection_regions(self):
        events = ak.Array({
            "nFatJet": [2, 2, 2, 1],
            "FatJet_isGood": [[True, True], [True, True], [True, False], [True, True]],
            "DeltaEtaJ0J1FatJet": [1.8, 1.8, 1.8, 1.8],
            "RTFatJet": [0.30, 0.20, 0.30, 0.30],
            "FatJet_pt": [[500.0, 400.0], [500.0, 400.0], [500.0, 400.0], [500.0]],
            "FatJet_photonEnergyFraction": [[0.1, 0.1], [0.1, 0.1], [0.1, 0.1], [0.1]],
            "run": [319000, 319000, 319000, 319000],
            "lumSec": [1, 2, 3, 4],
            "event": [10, 20, 30, 40],
        })
        mask_val = get_event_selection(events, "data", "2017", region="validation")
        # Event 0: passes (isGood, deta=1.8, RT=0.30)
        # Event 1: fails RT > 0.24
        # Event 2: fails isGood on j1
        # Event 3: fails nFatJet >= 2
        np.testing.assert_array_equal(mask_val, [True, False, False, False])

        mask_high_rt = get_event_selection(events, "data", "2017", region="highRT")
        np.testing.assert_array_equal(mask_high_rt, mask_val)

        mask_low_rt = get_event_selection(events, "data", "2017", region="lowRT")
        np.testing.assert_array_equal(mask_low_rt, [False, True, False, False])

        mask_ctrl = get_event_selection(events, "data", "2017", region="control")
        np.testing.assert_array_equal(mask_ctrl, [True, True, False, False])

        mask_inc = get_event_selection(events, "data", "2017", region="inclusive")
        np.testing.assert_array_equal(mask_inc, [True, True, False, False])

    def test_2018_hem_veto(self):
        # Event 0: passes HEMVeto
        # Event 1: fails HEMVeto
        events = ak.Array({
            "nFatJet": [2, 2],
            "FatJet_isGood": [[True, True], [True, True]],
            "DeltaEtaJ0J1FatJet": [1.8, 1.8],
            "RTFatJet": [0.30, 0.30],
            "FatJet_pt": [[350.0, 300.0], [350.0, 300.0]],
            "FatJet_photonEnergyFraction": [[0.1, 0.1], [0.1, 0.1]],
            "run": [319500, 319500],
            "lumSec": [1, 1],
            "event": [1, 1],
            "HEMVeto": [True, False],
        })
        mask_mc = get_event_selection(events, "background", "2018", region="validation", period="posthem")
        self.assertTrue(mask_mc[0])
        self.assertFalse(mask_mc[1])


class TestMajorBackgrounds(unittest.TestCase):
    def test_family_names(self):
        self.assertEqual(bkg_family_name("QCD_HT1000to1500"), "QCD")
        self.assertEqual(bkg_family_name("TTJets_TuneCP5"), "TTJets")
        self.assertEqual(bkg_family_name("TTJets_HT-800to1200"), "TTJets")
        self.assertEqual(bkg_family_name("WJetsToLNu_HT-600To800"), "WJets")
        self.assertEqual(bkg_family_name("ZJetsToNuNu_HT-1200To2500"), "ZJets")

    def test_family_grouping(self):
        groups = {
            "QCD_HT500to700": ["f1.root", "f2.root"],
            "QCD_HT1000to1500": ["f3.root"],
            "TTJets_TuneCP5": ["f4.root"],
        }
        fam_groups = bkg_family_groups(groups)
        self.assertIn("QCD", fam_groups)
        self.assertIn("TTJets", fam_groups)
        self.assertEqual(len(fam_groups["QCD"]), 2)
        self.assertEqual(len(fam_groups["TTJets"]), 1)


class TestTH1Accumulator(unittest.TestCase):
    def test_filling_and_addition(self):
        bins = np.linspace(0, 10, 6)  # 5 bins: [0, 2, 4, 6, 8, 10]
        h1 = TH1Accumulator(bins, name="test")
        h1.fill(np.array([1.0, 3.0, 5.0]), weights=np.array([1.0, 2.0, 3.0]))

        np.testing.assert_allclose(h1.counts, [1.0, 2.0, 3.0, 0.0, 0.0])
        np.testing.assert_allclose(h1.sumw2, [1.0, 4.0, 9.0, 0.0, 0.0])

        h2 = TH1Accumulator(bins, name="test")
        h2.fill(np.array([1.0, 7.0]), weights=np.array([2.0, 1.5]))
        h1.add(h2)

        np.testing.assert_allclose(h1.counts, [3.0, 2.0, 3.0, 1.5, 0.0])
        np.testing.assert_allclose(h1.sumw2, [5.0, 4.0, 9.0, 2.25, 0.0])


class TestRootIOAndPlotting(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_save_and_load_histograms(self):
        bins = np.linspace(650, 2000, 28)
        acc_qcd = TH1Accumulator(bins, name="MT01FatJetMET")
        acc_qcd.fill(np.array([700.0, 800.0]), weights=np.array([10.5, 5.2]))

        acc_data = TH1Accumulator(bins, name="MT01FatJetMET")
        acc_data.fill(np.array([700.0, 850.0]), weights=np.array([1.0, 1.0]))

        data_to_save = {
            "QCD": {"MT01FatJetMET": acc_qcd},
            "Data": {"MT01FatJetMET": acc_data},
        }

        root_path = os.path.join(self.temp_dir, "test_hist.root")
        save_histograms_to_root(data_to_save, "2018", "validation", root_path)
        self.assertTrue(os.path.isfile(root_path))

        loaded = load_histograms_from_root(root_path)
        self.assertIn("QCD", loaded)
        self.assertIn("Data", loaded)
        np.testing.assert_allclose(loaded["QCD"]["MT01FatJetMET"].counts, acc_qcd.counts)
        np.testing.assert_allclose(loaded["QCD"]["MT01FatJetMET"].sumw2, acc_qcd.sumw2)
        np.testing.assert_allclose(loaded["Data"]["MT01FatJetMET"].counts, acc_data.counts)

    def test_plot_stack_rendering(self):
        bins = VARIABLES["MT01FatJetMET"]["bins"]
        acc_qcd = TH1Accumulator(bins)
        acc_qcd.fill(np.array([700.0, 800.0, 900.0]), weights=np.array([10.0, 8.0, 4.0]))

        acc_tt = TH1Accumulator(bins)
        acc_tt.fill(np.array([750.0, 850.0]), weights=np.array([2.0, 1.5]))

        acc_data = TH1Accumulator(bins)
        acc_data.fill(np.array([700.0, 800.0]), weights=np.array([12.0, 9.0]))

        acc_sig = TH1Accumulator(bins)
        acc_sig.fill(np.array([1000.0, 1100.0]), weights=np.array([1.0, 1.5]))

        hist_dict = {
            "QCD": {"MT01FatJetMET": acc_qcd},
            "TTJets": {"MT01FatJetMET": acc_tt},
            "Data": {"MT01FatJetMET": acc_data},
            "s-channel_mMed-1000": {"MT01FatJetMET": acc_sig},
        }

        plot_out = os.path.join(self.temp_dir, "plots")
        plot_stacked_variable(
            "MT01FatJetMET",
            hist_dict,
            year="2018",
            region="validation",
            out_dir=plot_out,
        )

        pdf_path = os.path.join(plot_out, "mt_2018_validation.pdf")
        png_path = os.path.join(plot_out, "mt_2018_validation.png")
        self.assertTrue(os.path.isfile(pdf_path), f"Expected PDF file at {pdf_path}")
        self.assertFalse(os.path.isfile(png_path), f"PNG file should not be created")
        self.assertGreater(os.path.getsize(pdf_path), 1000)

    def test_cms_header_styling(self):
        import matplotlib.pyplot as plt
        from plotting.plot_style import add_cms_header

        fig, ax = plt.subplots()
        add_cms_header(ax, year="2018", has_data=True)
        title_data = ax.get_title(loc="left")
        self.assertIn("Preliminary", title_data)
        self.assertNotIn("Simulation", title_data)
        plt.close(fig)

        fig, ax = plt.subplots()
        add_cms_header(ax, year="2018", has_data=False)
        title_sim = ax.get_title(loc="left")
        self.assertIn("Simulation", title_sim)
        self.assertIn("Preliminary", title_sim)
        plt.close(fig)

    def test_format_yield(self):
        from plotting.plot_style import format_yield
        self.assertEqual(format_yield(0.0), "0")
        self.assertEqual(format_yield(123456.0), "1.23e+05")
        self.assertEqual(format_yield(1500.0), "1,500")
        self.assertEqual(format_yield(42.5), "42.5")
        self.assertEqual(format_yield(3.1415), "3.14")

    def test_export_yields_csv(self):
        from plotting.plot_stack import export_yields_csv
        bins = VARIABLES["MT01FatJetMET"]["bins"]
        acc_qcd = TH1Accumulator(bins)
        acc_qcd.fill(np.array([700.0, 800.0]), weights=np.array([10.0, 20.0]))
        acc_data = TH1Accumulator(bins)
        acc_data.fill(np.array([700.0, 800.0]), weights=np.array([12.0, 22.0]))
        acc_sig = TH1Accumulator(bins)
        acc_sig.fill(np.array([1000.0]), weights=np.array([5.0]))

        hist_dict = {
            "QCD": {"MT01FatJetMET": acc_qcd},
            "Data": {"MT01FatJetMET": acc_data},
            "s-channel_mMed-1000": {"MT01FatJetMET": acc_sig},
        }

        plot_out = os.path.join(self.temp_dir, "csv_test")
        csv_path = export_yields_csv(
            hist_dict,
            year="2018",
            region="validation",
            out_dir=plot_out,
            selected_vars=["MT01FatJetMET"],
        )
        self.assertTrue(os.path.isfile(csv_path))
        with open(csv_path) as f:
            content = f.read()
        self.assertIn("MT01FatJetMET", content)
        self.assertIn("QCD", content)
        self.assertIn("30.00", content)  # QCD yield = 10 + 20
        self.assertIn("34.00", content)  # Data yield = 12 + 22
        self.assertIn("s-channel_mMed-1000", content)

        tidy_csv = os.path.join(plot_out, "yields_per_process.csv")
        self.assertTrue(os.path.isfile(tidy_csv))

    def test_cutflow_csv_export(self):
        from plotting.cutflow import export_and_print_cutflow
        records = [
            {
                "name": "QCD",
                "type": "background",
                "root_cf": {"Initial": 1000000, "Trigger": 800000, "Final": 600000},
                "analysis_cf": {"Events_skim": 600000, "Good_j0_j1": 550000, "1.5 < |dEta| < 2.2": 80000},
                "analysis_weighted": {"Events_skim": 60000.0, "Good_j0_j1": 55000.0, "1.5 < |dEta| < 2.2": 8000.0},
            },
            {
                "name": "Data",
                "type": "data",
                "root_cf": {"Initial": 5000000, "Trigger": 4500000, "Final": 4000000},
                "analysis_cf": {"Events_skim": 4000000, "Good_j0_j1": 3900000, "1.5 < |dEta| < 2.2": 150000},
                "analysis_weighted": {"Events_skim": 4000000.0, "Good_j0_j1": 3900000.0, "1.5 < |dEta| < 2.2": 150000.0},
            },
        ]
        csv_file = os.path.join(self.temp_dir, "test_cutflow.csv")
        export_and_print_cutflow(records, "2018", "validation", csv_file)
        self.assertTrue(os.path.isfile(csv_file))

        with open(csv_file) as f:
            content = f.read()
        self.assertIn("ROOT/Initial", content)
        self.assertIn("ROOT/Trigger", content)
        self.assertIn("Analysis/Good_j0_j1", content)
        self.assertIn("Final_Raw_Pass", content)
        self.assertIn("Final_Weighted_Yield", content)
        self.assertIn("QCD", content)
        self.assertIn("Data", content)
        self.assertIn("8000.00", content)


class TestCLIRegionArgs(unittest.TestCase):
    def test_run_plots_region_parsing(self):
        import sys
        from plotting.run_plots import parse_args
        orig_argv = sys.argv
        try:
            sys.argv = ["run_plots.py", "--region", "control", "lowRT"]
            args = parse_args()
            self.assertEqual(args.region, ["control", "lowRT"])

            sys.argv = ["run_plots.py", "--regions", "highRT"]
            args = parse_args()
            self.assertEqual(args.region, ["highRT"])
        finally:
            sys.argv = orig_argv

    def test_cutflow_region_parsing(self):
        import sys
        from plotting.cutflow import parse_args
        orig_argv = sys.argv
        try:
            sys.argv = ["cutflow.py", "--region", "control", "inclusive"]
            args = parse_args()
            self.assertEqual(args.region, ["control", "inclusive"])
        finally:
            sys.argv = orig_argv


if __name__ == "__main__":
    unittest.main()



