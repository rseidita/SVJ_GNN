import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

SVJ_GNN_DIR = Path(__file__).resolve().parent.parent


class TestResumeTraining(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(prefix="svj_test_resume_")
        self.out_dir = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_checkpoint_payload_structure(self):
        """Verify the checkpoint structure contains all required persistence keys."""
        run_name = "test_payload"
        cmd = [
            sys.executable,
            str(SVJ_GNN_DIR / "train_gae.py"),
            "--train-on", "smoke",
            "--epochs", "1",
            "--device", "cpu",
            "--no-comet",
            "--output-dir", str(self.out_dir),
            "--run-name", run_name,
            "--quiet",
        ]
        result = subprocess.run(cmd, cwd=SVJ_GNN_DIR, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, f"Training failed:\n{result.stderr}\n{result.stdout}")

        ckpt_path = self.out_dir / run_name / "checkpoint.pt"
        latest_path = self.out_dir / run_name / "checkpoint_latest.pt"
        self.assertTrue(ckpt_path.exists())
        self.assertTrue(latest_path.exists())

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.assertIn("state_dict", ckpt)
        self.assertIn("optimizer_state_dict", ckpt)
        self.assertIn("epoch", ckpt)
        self.assertEqual(ckpt["epoch"], 1)
        self.assertIn("history", ckpt)
        self.assertIn("best_val_loss", ckpt)
        self.assertIn("means", ckpt)
        self.assertIn("stds", ckpt)
        self.assertIn("split_state", ckpt)
        self.assertEqual(len(ckpt["history"]["epoch_train"]), 1)

    def test_resume_extends_training_and_history(self):
        """Train 1 epoch, resume to 2 total epochs, verify history has 2 epochs."""
        run_name = "test_resume_extend"
        cmd_init = [
            sys.executable,
            str(SVJ_GNN_DIR / "train_gae.py"),
            "--train-on", "smoke",
            "--epochs", "1",
            "--device", "cpu",
            "--no-comet",
            "--output-dir", str(self.out_dir),
            "--run-name", run_name,
            "--quiet",
        ]
        res_init = subprocess.run(cmd_init, cwd=SVJ_GNN_DIR, capture_output=True, text=True)
        self.assertEqual(res_init.returncode, 0, f"Init training failed:\n{res_init.stderr}")

        ckpt_path = self.out_dir / run_name / "checkpoint.pt"

        # Resume to 2 total epochs
        cmd_resume = [
            sys.executable,
            str(SVJ_GNN_DIR / "train_gae.py"),
            "--train-on", "smoke",
            "--resume", str(ckpt_path),
            "--epochs", "2",
            "--device", "cpu",
            "--no-comet",
            "--output-dir", str(self.out_dir),
            "--run-name", run_name,
            "--quiet",
        ]
        res_resume = subprocess.run(cmd_resume, cwd=SVJ_GNN_DIR, capture_output=True, text=True)
        self.assertEqual(res_resume.returncode, 0, f"Resume failed:\n{res_resume.stderr}\n{res_resume.stdout}")

        ckpt_resumed = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.assertEqual(ckpt_resumed["epoch"], 2)
        self.assertEqual(len(ckpt_resumed["history"]["epoch_train"]), 2)

        history_file = self.out_dir / run_name / "history.json"
        with open(history_file) as f:
            hist = json.load(f)
        self.assertEqual(len(hist["epoch_train"]), 2)

    def test_resume_with_add_epochs(self):
        """Train 1 epoch, then use --add-epochs 1 to reach 2 total epochs."""
        run_name = "test_add_epochs"
        cmd_init = [
            sys.executable,
            str(SVJ_GNN_DIR / "train_gae.py"),
            "--train-on", "smoke",
            "--epochs", "1",
            "--device", "cpu",
            "--no-comet",
            "--output-dir", str(self.out_dir),
            "--run-name", run_name,
            "--quiet",
        ]
        res_init = subprocess.run(cmd_init, cwd=SVJ_GNN_DIR, capture_output=True, text=True)
        self.assertEqual(res_init.returncode, 0, f"Init training failed:\n{res_init.stderr}")

        ckpt_path = self.out_dir / run_name / "checkpoint.pt"

        cmd_add = [
            sys.executable,
            str(SVJ_GNN_DIR / "train_gae.py"),
            "--train-on", "smoke",
            "--resume", str(ckpt_path),
            "--add-epochs", "1",
            "--device", "cpu",
            "--no-comet",
            "--output-dir", str(self.out_dir),
            "--run-name", run_name,
            "--quiet",
        ]
        res_add = subprocess.run(cmd_add, cwd=SVJ_GNN_DIR, capture_output=True, text=True)
        self.assertEqual(res_add.returncode, 0, f"Add epochs failed:\n{res_add.stderr}")

        ckpt_resumed = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.assertEqual(ckpt_resumed["epoch"], 2)
        self.assertEqual(len(ckpt_resumed["history"]["epoch_train"]), 2)

    def test_legacy_checkpoint_tolerance(self):
        """Simulate a legacy checkpoint without optimizer_state_dict and epoch."""
        run_name = "test_legacy"
        cmd_init = [
            sys.executable,
            str(SVJ_GNN_DIR / "train_gae.py"),
            "--train-on", "smoke",
            "--epochs", "1",
            "--device", "cpu",
            "--no-comet",
            "--output-dir", str(self.out_dir),
            "--run-name", run_name,
            "--quiet",
        ]
        subprocess.run(cmd_init, cwd=SVJ_GNN_DIR, check=True)

        ckpt_path = self.out_dir / run_name / "checkpoint.pt"
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        # Strip optimizer state, epoch, and history to simulate legacy checkpoint
        del ckpt["optimizer_state_dict"]
        del ckpt["epoch"]
        del ckpt["history"]
        torch.save(ckpt, ckpt_path)

        cmd_resume = [
            sys.executable,
            str(SVJ_GNN_DIR / "train_gae.py"),
            "--train-on", "smoke",
            "--resume", str(ckpt_path),
            "--epochs", "1",
            "--device", "cpu",
            "--no-comet",
            "--output-dir", str(self.out_dir),
            "--run-name", run_name,
            "--quiet",
        ]
        res_resume = subprocess.run(cmd_resume, cwd=SVJ_GNN_DIR, capture_output=True, text=True)
        self.assertEqual(res_resume.returncode, 0, f"Legacy resume failed:\n{res_resume.stderr}\n{res_resume.stdout}")


if __name__ == "__main__":
    unittest.main()

