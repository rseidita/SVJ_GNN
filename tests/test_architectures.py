import unittest

import torch
from torch_geometric.data import Batch, Data
from torch_geometric.utils import batched_negative_sampling

from architectures.GraphTransformerAE import GraphTransformerAE


class TransformerAutoencoderTests(unittest.TestCase):
    @staticmethod
    def graphs():
        return [
            Data(
                x=torch.randn(4, 5),
                edge_index=torch.tensor([
                    [0, 1, 1, 2, 2, 3],
                    [1, 0, 2, 1, 3, 2],
                ]),
            ),
            Data(
                x=torch.randn(3, 5),
                edge_index=torch.tensor([
                    [0, 1, 1, 2],
                    [1, 0, 2, 1],
                ]),
            ),
        ]

    def test_training_losses_are_finite_and_differentiable(self):
        torch.manual_seed(7)
        batch = Batch.from_data_list(self.graphs())
        model = GraphTransformerAE(
            num_features=5,
            architecture=(12, 6),
            heads=2,
            dropout=0.1,
        )
        model.train()
        negative_edges = batched_negative_sampling(
            batch.edge_index, batch.batch
        )
        loss = (
            model.edges_loss(
                batch.x, batch.edge_index, negative_edges
            )
            + model.features_loss(batch.x, batch.edge_index)
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(any(
            parameter.grad is not None
            and torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        ))

    def test_forward_and_scoring_match_single_graph_interface(self):
        torch.manual_seed(11)
        graphs = self.graphs()
        batch = Batch.from_data_list(graphs)
        model = GraphTransformerAE(
            num_features=5,
            architecture=(8, 4),
            heads=2,
        )
        model.eval()

        with torch.inference_mode():
            reconstructed, edge_probabilities = model(
                batch.x, batch.edge_index
            )
            batch_scores = model.score_batch(
                batch.x, batch.edge_index, batch.batch, alpha=0.5
            )
            event_scores = model.score_event(graphs[0], alpha=0.5)

        self.assertEqual(reconstructed.shape, batch.x.shape)
        self.assertEqual(
            edge_probabilities.shape, (batch.edge_index.shape[1],)
        )
        for component in batch_scores:
            self.assertEqual(component.shape, (2,))
            self.assertTrue(torch.isfinite(component).all())
        for batched, individual in zip(batch_scores, event_scores):
            self.assertTrue(
                torch.allclose(
                    batched[0], individual, rtol=1e-5, atol=1e-6
                )
            )


if __name__ == "__main__":
    unittest.main()
