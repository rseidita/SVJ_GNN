import os
import tempfile
import unittest
from pathlib import Path

import awkward as ak
import torch
import uproot
from torch_geometric.data import Batch, Data

from architectures.ModelRegistry import build_model, validate_model
from data_loading.graphDataLoader import GraphDataLoader
from utils.GaeConfig import load_config
from utils.JetGraphProducer import JetGraphProducer
from utils.LundTreeUtilities import (
    OnTheFlyNormalizer,
    compute_normalization_stats,
)


def write_fixture(root: Path):
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


class TestAimarModel(unittest.TestCase):
    @staticmethod
    def dummy_graphs(num_features=12):
        return [
            Data(
                x=torch.randn(4, num_features),
                edge_index=torch.tensor([
                    [0, 1, 1, 2, 2, 3],
                    [1, 0, 2, 1, 3, 2],
                ], dtype=torch.long),
            ),
            Data(
                x=torch.randn(3, num_features),
                edge_index=torch.tensor([
                    [0, 1, 1, 2],
                    [1, 0, 2, 1],
                ], dtype=torch.long),
            ),
        ]

    def test_build_and_validate_model_registry(self):
        model = build_model(
            "GraphAEBatchNorm.GraphAEBatchNorm",
            num_features=12,
            architecture=[115, 170, 200, 115],
            dropout=0.1,
        )
        validate_model(model, "GraphAEBatchNorm.GraphAEBatchNorm")
        self.assertIsNotNone(model)

    def test_state_dict_exact_match_with_aimar_checkpoint(self):
        repo_root = Path(__file__).resolve().parent.parent.parent
        ckpt_path = repo_root / "Aimar_models" / "artifacts" / "aimar_graph_autoencoder.pt"
        if not ckpt_path.is_file():
            self.skipTest(f"Checkpoint not found at {ckpt_path}")

        saved_state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model = build_model(
            "GraphAEBatchNorm.GraphAEBatchNorm",
            num_features=12,
            architecture=[115, 170, 200, 115],
            dropout=0.1,
        )
        load_result = model.load_state_dict(saved_state, strict=True)
        self.assertEqual(len(load_result.missing_keys), 0)
        self.assertEqual(len(load_result.unexpected_keys), 0)

    def test_training_loss_and_gradients(self):
        torch.manual_seed(42)
        model = build_model(
            "GraphAEBatchNorm.GraphAEBatchNorm",
            num_features=12,
            architecture=[115, 170, 200, 115],
            dropout=0.1,
        )
        model.train()
        graphs = self.dummy_graphs(12)
        batch = Batch.from_data_list(graphs)

        total, edges, features = model.training_loss(batch, alpha=0.2)
        self.assertTrue(torch.isfinite(total))
        self.assertTrue(torch.isfinite(edges))
        self.assertTrue(torch.isfinite(features))

        total.backward()
        has_grads = any(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in model.parameters()
        )
        self.assertTrue(has_grads)

    def test_forward_and_scoring_interface(self):
        torch.manual_seed(42)
        model = build_model(
            "GraphAEBatchNorm.GraphAEBatchNorm",
            num_features=12,
            architecture=[115, 170, 200, 115],
            dropout=0.1,
        )
        model.eval()
        graphs = self.dummy_graphs(12)
        batch = Batch.from_data_list(graphs)

        with torch.no_grad():
            reconstructed, edge_prob = model(batch.x, batch.edge_index)
            self.assertEqual(reconstructed.shape, batch.x.shape)
            self.assertEqual(edge_prob.shape, (batch.edge_index.shape[1],))

            total, edge_score, feature_score = model.score_batch(
                batch.x, batch.edge_index, batch.batch, alpha=0.2
            )
            self.assertEqual(total.shape, (2,))
            self.assertEqual(edge_score.shape, (2,))
            self.assertEqual(feature_score.shape, (2,))

            ev_total, ev_edge, ev_feat = model.score_event(graphs[0], alpha=0.2)
            self.assertTrue(torch.isfinite(ev_total))
            self.assertTrue(torch.isfinite(ev_edge))
            self.assertTrue(torch.isfinite(ev_feat))

    def test_aimar_config_parsing(self):
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "gae_config_aimar.json",
        )
        config = load_config(config_path)
        self.assertTrue(config.graph["use_lund_decomp"])
        self.assertEqual(config.graph["n_lund_vars"], 5)
        self.assertTrue(config.graph["save_4vectors_in_lund_tree"])
        self.assertTrue(config.graph["save_energy_fractions"])
        self.assertEqual(config.training["alpha"], 0.2)
        self.assertEqual(config.model.spec, "GraphAEBatchNorm.GraphAEBatchNorm")
        self.assertEqual(config.model.kwargs["architecture"], [115, 170, 200, 115])

    def test_production_and_pipeline_with_aimar_graph_settings(self):
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "gae_config_aimar.json",
        )
        config = load_config(config_path)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            write_fixture(tmp_path)

            # Produce graphs with config.graph settings
            producer = JetGraphProducer(
                root=str(tmp_path),
                output_dataset_name="aimar_fixture",
                n_store_jets=config.graph.get("n_store_jets", 2),
                use_lund_decomp=config.graph.get("use_lund_decomp", True),
                n_lund_vars=config.graph.get("n_lund_vars", 5),
                save_energy_fractions=config.graph.get("save_energy_fractions", True),
                save_4vectors_in_lund_tree=config.graph.get("save_4vectors_in_lund_tree", True),
                use_displacement=config.graph.get("use_displacement", False),
                kt_cut=config.graph.get("kt_cut", None),
                verbose=False,
            )

            self.assertGreater(len(producer), 0)
            graph = producer[0]
            # Must have exactly 12 node features
            self.assertEqual(graph.x.shape[1], 12)
            self.assertEqual(graph.num_node_features, 12)

            # Test DataLoader and training pass with GraphAEBatchNorm
            dataset_paths = [[str(tmp_path), "aimar_fixture"]]
            loader = GraphDataLoader(
                dataset_paths=dataset_paths,
                sampling="sequential",
                batch_size=4,
            )
            means, stds = compute_normalization_stats(loader, 12, verbose=False)
            self.assertEqual(means.shape, (12,))
            self.assertEqual(stds.shape, (12,))

            normalizer = OnTheFlyNormalizer(["x"], means, stds)
            norm_loader = GraphDataLoader(
                dataset_paths=dataset_paths,
                sampling="sequential",
                normalizer=normalizer,
                batch_size=4,
            )

            model = build_model(
                config.model.spec,
                num_features=12,
                **config.model.kwargs,
            )
            model.train()
            batch = next(iter(norm_loader))
            total, edges, features = model.training_loss(batch, alpha=config.training["alpha"])
            self.assertTrue(torch.isfinite(total))
            total.backward()


if __name__ == "__main__":
    unittest.main()

