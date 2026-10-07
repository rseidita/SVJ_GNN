import os
import sys
import unittest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from architectures.ModelRegistry import build_model
from utils.GaeConfig import load_config
import condor.submit_condor as submit_condor


class TestConfigPathsAndSeeding(unittest.TestCase):
    def setUp(self):
        self.gnn_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.aimar_config_path = os.path.join(self.gnn_dir, "gae_config_aimar.json")
        self.default_config_path = os.path.join(self.gnn_dir, "gae_config.json")

    def test_config_output_directories_parsed(self):
        aimar_cfg = load_config(self.aimar_config_path)
        self.assertEqual(aimar_cfg.production.get("output_dir"), "data_aimar")
        self.assertEqual(aimar_cfg.training.get("output_dir"), "training_aimar")
        self.assertEqual(aimar_cfg.scoring.get("output_dir"), "output_scores_aimar")
        self.assertEqual(aimar_cfg.plotting.get("plots_dir"), "plots_aimar")

        default_cfg = load_config(self.default_config_path)
        self.assertEqual(default_cfg.production.get("output_dir"), "data")
        self.assertEqual(default_cfg.training.get("output_dir"), "training")
        self.assertEqual(default_cfg.scoring.get("output_dir"), "output_scores")
        self.assertEqual(default_cfg.plotting.get("plots_dir"), "plots")

    def test_dataset_root_uses_config_output_dir(self):
        aimar_cfg = load_config(self.aimar_config_path)
        sample = aimar_cfg.sample("data_2018")
        root = aimar_cfg.dataset_root(sample)
        self.assertEqual(root, "data_aimar/data_2018")

    def test_model_initialization_seeding(self):
        torch.manual_seed(42)
        model1 = build_model(
            "GraphAEBatchNorm.GraphAEBatchNorm",
            num_features=12,
            architecture=[115, 170, 200, 115],
        )

        torch.manual_seed(42)
        model2 = build_model(
            "GraphAEBatchNorm.GraphAEBatchNorm",
            num_features=12,
            architecture=[115, 170, 200, 115],
        )

        torch.manual_seed(12345)
        model3 = build_model(
            "GraphAEBatchNorm.GraphAEBatchNorm",
            num_features=12,
            architecture=[115, 170, 200, 115],
        )

        # model1 and model2 with seed 42 must have identical parameters
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            self.assertTrue(torch.allclose(p1, p2))

        # model1 and model3 with seed 42 vs 12345 must have different parameters
        differs = False
        for p1, p3 in zip(model1.parameters(), model3.parameters()):
            if not torch.allclose(p1, p3):
                differs = True
                break
        self.assertTrue(differs, "Models with different seeds should have different initial weights")

    def test_condor_queue_items_multi_seed(self):
        parser = submit_condor.parse_arguments
        args = parser.__wrapped__ if hasattr(parser, "__wrapped__") else None
        # Test queue_items directly with mock args
        class Args:
            stage = "train"
            seeds = [42, 123, 999]
            seed = None
            config = "gae_config_aimar.json"
            files_per_job = 5
            job_arguments = ""

        items = submit_condor.queue_items(Args(), None)
        self.assertEqual(
            items,
            [
                "train, train_seed42, 0",
                "train, train_seed123, 0",
                "train, train_seed999, 0",
            ],
        )

    def test_condor_queue_items_single_seed(self):
        class Args:
            stage = "train"
            seeds = None
            seed = 42
            config = "gae_config_aimar.json"
            files_per_job = 5
            job_arguments = ""

        items = submit_condor.queue_items(Args(), None)
        self.assertEqual(items, ["train, train_seed42, 0"])
        args_str = submit_condor.job_arguments(Args())
        self.assertIn("--seed 42", args_str)


if __name__ == "__main__":
    unittest.main()
