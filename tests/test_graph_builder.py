import tempfile
import unittest
from pathlib import Path

import awkward as ak
import numpy as np
import torch
import uproot

from utils.GraphBuilder import GraphBuildConfig, build_pfnano_jet_graph
from utils.LundTreeUtilities import (
    get_lund_decomp,
    prune_tree,
    tensor_to_tree,
    tree_to_tensor,
)


class GraphBuilderTests(unittest.TestCase):
    def test_lund_feature_layouts_and_tree_invariants(self):
        for n_lund_vars in (3, 5):
            for save_4vectors in (False, True):
                for save_fractions in (False, True):
                    for use_displacement in (False, True):
                        with self.subTest(
                            n_lund_vars=n_lund_vars,
                            save_4vectors=save_4vectors,
                            save_fractions=save_fractions,
                            use_displacement=use_displacement,
                        ):
                            n = 8
                            fractions = np.vstack([
                                np.zeros((3, n)),
                                np.ones(n),
                            ])
                            nodes, edges = get_lund_decomp(
                                np.linspace(10.0, 1.0, n),
                                np.linspace(-0.2, 0.2, n),
                                np.linspace(-3.1, 3.1, n),
                                np.zeros(n),
                                np.zeros(n),
                                np.zeros(n),
                                fractions,
                                save_fractions,
                                n_lund_vars,
                                save_4vectors,
                                use_displacement,
                            )
                            self.assertEqual(nodes.shape[1], 2 * n - 1)
                            self.assertEqual(edges.shape, (2, 4 * (n - 1)))
                            self.assertEqual(
                                np.unique(edges, axis=1).shape[1],
                                edges.shape[1],
                            )
                            root = tensor_to_tree(nodes.T, edges)
                            roundtrip_nodes, roundtrip_edges = tree_to_tensor(
                                root, nodes.T.shape
                            )
                            self.assertEqual(
                                roundtrip_nodes.shape, nodes.T.shape
                            )
                            self.assertEqual(
                                roundtrip_edges.shape, edges.shape
                            )

                            pruned_root, n_pruned_nodes = prune_tree(
                                tensor_to_tree(nodes.T, edges),
                                0,
                                float(np.median(nodes[0])),
                            )
                            _, pruned_edges = tree_to_tensor(
                                pruned_root,
                                (n_pruned_nodes, nodes.shape[0]),
                            )
                            self.assertEqual(
                                pruned_edges.shape[1],
                                2 * (n_pruned_nodes - 1),
                            )

    def test_single_constituent_lund_tree(self):
        fractions = np.asarray([[0.0], [0.0], [0.0], [1.0]])
        nodes, edges = get_lund_decomp(
            np.asarray([10.0]),
            np.asarray([0.0]),
            np.asarray([0.0]),
            np.asarray([0.14]),
            np.asarray([0.0]),
            np.asarray([0.0]),
            fractions,
        )
        self.assertEqual(nodes.shape, (3, 1))
        self.assertEqual(edges.shape, (2, 0))

    def test_spatial_graph_wraps_phi_boundary(self):
        with tempfile.TemporaryDirectory(prefix="svj_graph_test_") as directory:
            file_name = Path(directory) / "fixture.root"
            event = ak.Array([{
                "nFatJet": 1,
                "FatJet_pt": [100.0],
                "FatJet_eta": [0.1],
                "FatJet_phi": [3.13],
                "FatJetPFCands_jetIdx": [0, 0],
                "FatJetPFCands_pFCandsIdx": [0, 1],
                "PFCands_pt": [30.0, 20.0],
                "PFCands_eta": [0.11, 0.08],
                "PFCands_phi": [3.12, -3.13],
                "PFCands_pdgId": [211, 22],
                "PFCands_mass": [0.14, 0.0],
            }])
            with uproot.recreate(file_name) as output:
                output["Events"] = event
            with uproot.open(f"{file_name}:Events") as tree:
                loaded_event = tree.arrays(library="ak")[0]

            graph, _ = build_pfnano_jet_graph(
                loaded_event,
                0,
                GraphBuildConfig(
                    use_lund_decomp=False,
                    delta_r_threshold=0.2,
                ),
            )
            edges = {tuple(edge) for edge in graph.edge_index.T.tolist()}
            self.assertIn((0, 1), edges)
            self.assertIn((1, 0), edges)
            self.assertTrue(torch.isfinite(graph.x).all())


if __name__ == "__main__":
    unittest.main()
