from collections import deque

import numpy as np
import torch
from torch_geometric.transforms import BaseTransform
from tqdm import tqdm
import numba as nb

# Avoid nasty interplays of torch and numba
torch.multiprocessing.set_sharing_strategy('file_system')

class OnTheFlyNormalizer(BaseTransform):
    """
    Normalize the data on the fly by simple subtraction and division.

    Args:
        attrs: list of attributes to normalize
        means: means of the attributes
        stds: standard deviations of the attributes
    """
    def __init__(self, attrs: list, means: torch.tensor, stds: torch.tensor, device=None):
        self.attrs = attrs
        self.means = means.to(device) if device is not None else means
        self.stds = stds.to(device) if device is not None else stds
        
    def __call__(self, data):
        for store in data.stores:
            for key, value in store.items(*self.attrs):
                value -= self.means
                value /= self.stds
                store[key] = value

        return data


def _node_weights(batch):
    """Per-node event weights, falling back to unit weights.

    Graphs produced with ``weights=None``, which is the only option for data
    samples and for files without a Metadata tree, carry no ``w`` attribute.
    """
    weights = getattr(batch, "w", None)
    if weights is None:
        return torch.ones(batch.x.shape[0], dtype=torch.float64)
    return weights.reshape(-1)[batch.batch].to(torch.float64)


def compute_normalization_stats(loader, n_features, verbose=False):
    """Event-weighted mean and standard deviation of the node features.

    Statistics are accumulated over whatever partition *loader* selects, so
    pass a loader restricted to the training partition. The returned tensors
    are ready for ``OnTheFlyNormalizer(["x"], means, stds)``.
    """
    feature_sum = torch.zeros(n_features, dtype=torch.float64)
    total_node_weight = torch.tensor(0.0, dtype=torch.float64)

    for batch in tqdm(
        loader, desc="Computing training means", disable=not verbose
    ):
        node_weights = _node_weights(batch)
        features = batch.x.to(torch.float64)
        feature_sum += (features * node_weights[:, None]).sum(dim=0)
        total_node_weight += node_weights.sum()

    if total_node_weight <= 0:
        raise ValueError(
            "Total node weight is not positive; cannot compute normalization"
        )
    means = feature_sum / total_node_weight

    variance_sum = torch.zeros_like(means)
    for batch in tqdm(
        loader,
        desc="Computing training standard deviations",
        disable=not verbose,
    ):
        node_weights = _node_weights(batch)
        features = batch.x.to(torch.float64)
        variance_sum += (
            (features - means).square() * node_weights[:, None]
        ).sum(dim=0)

    stds = torch.sqrt(variance_sum / total_node_weight)
    # A constant feature would otherwise divide by zero.
    stds[stds == 0] = 1.0
    return means.float(), stds.float()

# Define binary tree structure for Lund decomposition
class Node:
    def __init__(self, value, index):
        self.value = value
        self.left = None
        self.right = None
        self.idx = index

def tensor_to_tree(nodes, edges):
    """
    Convert a binary tree from arrays to a tree structure for easier pruning.
    
    Args:
        nodes: array of node features
        edges: array of edges in the tree
        
    Returns:
        top_node: the top node of the tree
    """

    nodes = np.array(nodes)
    edges = np.array(edges)

    n_nodes = len(nodes)
    if n_nodes == 0:
        raise ValueError("Cannot convert an empty Lund tree")

    # Clustering appends each parent after its children, so the root has the
    # largest node index. A one-node tree has no edges.
    top_node_index = n_nodes - 1
    top_node = Node(nodes[top_node_index], top_node_index)

    neighbors = [[] for _ in range(n_nodes)]
    for source, target in zip(edges[0], edges[1]):
        source = int(source)
        target = int(target)
        if target not in neighbors[source]:
            neighbors[source].append(target)

    buffer = deque([(top_node, -1)])
    visited_nodes = {top_node_index}
    while buffer:
        node, parent_index = buffer.popleft()
        node_children = [
            idx for idx in neighbors[node.idx]
            if idx != parent_index and idx not in visited_nodes
        ]

        # Check that the tree is binary as it should be
        if len(node_children) not in [0, 2]:
            raise ValueError("Lund Tree is not binary!")
        if len(node_children) == 2:
            node.left = Node(nodes[node_children[0]], node_children[0])
            node.right = Node(nodes[node_children[1]], node_children[1])
            visited_nodes.update(node_children)
            buffer.append((node.left, node.idx))
            buffer.append((node.right, node.idx))

    if len(visited_nodes) != n_nodes:
        raise ValueError("Lund Tree is disconnected!")

    return top_node

def tree_to_tensor(top_node, shape):
    # Walk along the tree and convert it back into arrays
    nodes = np.zeros(shape)
    edges = np.empty((2, 2 * max(shape[0] - 1, 0)), dtype=np.int64)
    edge_offset = 0
    buffer = [top_node]
    top_node.idx = 0
    counter = 0
    nodes[0] = top_node.value

    while buffer:
        node = buffer.pop()

        if node.left is not None and node.right is not None:
            counter += 2
            node.left.idx = counter - 1
            node.right.idx = counter

            nodes[node.left.idx] = node.left.value
            nodes[node.right.idx] = node.right.value

            buffer.append(node.left)
            buffer.append(node.right)

            # Add both directions of each parent-child connection.
            edges[0, edge_offset:edge_offset + 4] = (
                node.idx, node.idx, node.left.idx, node.right.idx
            )
            edges[1, edge_offset:edge_offset + 4] = (
                node.left.idx, node.right.idx, node.idx, node.idx
            )
            edge_offset += 4

    return (
        torch.as_tensor(nodes, dtype=torch.float),
        torch.as_tensor(edges[:, :edge_offset], dtype=torch.long),
    )

def prune_tree(top_node, cut_variable_idx, cut_value):

    # Walk along the tree and prune nodes that do not pass the cut
    buffer = deque([top_node])
    n_nodes = 0
    while buffer:
        # Get the next node
        node = buffer.popleft()

        # Count the number of nodes
        n_nodes += 1

        # Check if the node passes the cut
        if node.value[cut_variable_idx] < cut_value:
            # Prune the node
            node.left = None
            node.right = None
        else:
            # Add the children to the buffer
            if node.left is not None:
                buffer.append(node.left)
            if node.right is not None:
                buffer.append(node.right)

    return top_node, n_nodes


@nb.njit(cache=True)
def get_rapidity(pt, eta, mass):
    return np.log((np.sqrt(mass**2 + (pt**2)*(np.cosh(eta)**2)) + pt*np.sinh(eta)) / np.sqrt(mass**2 + pt**2) + 1e-15)

@nb.njit(cache=True)
def get_delta(rapidity_a, rapidity_b, phi_a, phi_b):
    delta_phi = (phi_a - phi_b + np.pi) % (2 * np.pi) - np.pi
    return np.hypot(delta_phi, rapidity_a - rapidity_b)

@nb.njit(cache=True)
def get_z(pt_a, pt_b):
    return min(pt_a, pt_b) / (pt_a + pt_b)

@nb.njit(cache=True)
def get_psi(rapidity_low, rapidity_high, phi_low, phi_high):
    delta_phi = (phi_low - phi_high + np.pi) % (2 * np.pi) - np.pi
    return np.arctan2(rapidity_low - rapidity_high, delta_phi)

@nb.njit(cache=True)
def get_kt(pt_low, pt_high, delta):
    return min(pt_low, pt_high) * delta

@nb.njit(cache=True)
def get_energy(pt, eta, mass):
    return np.sqrt(mass**2 + (pt * np.cosh(eta))**2)

@nb.njit(cache=True)
def get_dij(constituents:np.ndarray)->np.ndarray:
    """
    Calculate the dij matrix for the Cambridge-Aachen clustering algorithm

    Args:
        constituents (np.ndarray): Array of constituents with shape (4, n_constituents)

    Returns:
        np.ndarray: dij matrix with shape (n_constituents, n_constituents)
    """
    y = get_rapidity(constituents[0], constituents[1], constituents[3])
    phi = constituents[2]
    n_constituents = len(y)
    dij = np.empty((n_constituents, n_constituents), dtype=np.float64)
    for i in range(n_constituents):
        dij[i, i] = np.inf
        for j in range(i + 1, n_constituents):
            delta_phi = (phi[i] - phi[j] + np.pi) % (2 * np.pi) - np.pi
            distance = (delta_phi**2 + (y[i] - y[j])**2) / 0.64
            dij[i, j] = distance
            dij[j, i] = distance

    return dij

@nb.njit(cache=True)
def cluster_constituents(constituents:np.ndarray, index_a:int, index_b:int):
    """Add two ``(pt, eta, phi, mass)`` four-vectors.

    This is implemented directly rather than through ``vector.obj`` because
    vector's object backend is not reliably compatible with Numba's nopython
    mode across vector/Numba version combinations.
    """
    pt_a = constituents[0, index_a]
    eta_a = constituents[1, index_a]
    phi_a = constituents[2, index_a]
    mass_a = constituents[3, index_a]

    pt_b = constituents[0, index_b]
    eta_b = constituents[1, index_b]
    phi_b = constituents[2, index_b]
    mass_b = constituents[3, index_b]

    px = pt_a * np.cos(phi_a) + pt_b * np.cos(phi_b)
    py = pt_a * np.sin(phi_a) + pt_b * np.sin(phi_b)
    pz = pt_a * np.sinh(eta_a) + pt_b * np.sinh(eta_b)

    energy_a = np.sqrt(mass_a**2 + (pt_a * np.cosh(eta_a))**2)
    energy_b = np.sqrt(mass_b**2 + (pt_b * np.cosh(eta_b))**2)
    energy = energy_a + energy_b

    pt = np.hypot(px, py)
    phi = np.arctan2(py, px)
    eta = np.arcsinh(pz / max(pt, 1e-15))
    mass_squared = energy**2 - px**2 - py**2 - pz**2
    mass = np.sqrt(max(mass_squared, 0.0))

    return pt, eta, phi, mass


@nb.njit(cache=True)
def get_lund_decomp(
        pt:np.ndarray,
        eta:np.ndarray,
        phi:np.ndarray,
        mass:np.ndarray,
        d0:np.ndarray, 
        dz:np.ndarray, 
        fractions:np.ndarray,
        save_fractions:bool=False,
        n_lund_vars:int=3,
        save_4vectors:bool=False,
        use_displacement:bool = False,
        ):
    """
    Perform jet declustering, returning the Lund tree as a nested list
    A nested list is probably not ideal, a willing student could improve it :)

    Args:       
        pt (np.ndarray): pt of jet constituents
        eta (np.ndarray): eta of jet constituents
        phi (np.ndarray): phi of jet constituents
        mass (np.ndarray): mass of jet constituents
        fractions (dict): energy fractions of jet constituents (basically pdgIds at this level)
        save_fractions (bool): whether to save the energy fractions of the constituents in the Lund tree
        n_lund_vars (int): number of Lund variables to use (3 or 5)
        save_constituents_4vectors (bool): whether to save the 4-vectors of the constituents in the Lund tree

    Returns:   
        tuple[list, list]: nodes (feature matrix), edges (sparse adjacency matrix)
    """

    if n_lund_vars not in [3, 5]:
        raise ValueError("Only 3 or 5 Lund variables are supported")

    n_initial = len(pt)
    if n_initial == 0:
        raise ValueError("Cannot build a Lund tree without constituents")

    constituents = np.stack((pt, eta, phi, mass), axis=0)

    if save_fractions:
        e_fraction = np.empty(2 * n_initial - 1, dtype=np.float64)
        mu_fraction = np.empty(2 * n_initial - 1, dtype=np.float64)
        g_fraction = np.empty(2 * n_initial - 1, dtype=np.float64)
        h_fraction = np.empty(2 * n_initial - 1, dtype=np.float64)
        e_fraction[:n_initial] = fractions[0]
        mu_fraction[:n_initial] = fractions[1]
        g_fraction[:n_initial] = fractions[2]
        h_fraction[:n_initial] = fractions[3]

    # Each of the n-1 merges adds two parent-child connections in both
    # directions. Preallocation avoids an array copy at every merge.
    edges = np.empty((2, 4 * max(n_initial - 1, 0)), dtype=np.int64)
    edge_offset = 0

    pt = pt.astype(np.float64)
    eta = eta.astype(np.float64)
    phi = phi.astype(np.float64)
    mass = mass.astype(np.float64)

    # A full binary clustering tree has exactly 2*n-1 nodes. Allocate feature
    # storage once rather than copying every feature array at each merge.
    n_tree_nodes = 2 * n_initial - 1
    nodes_pt = np.empty(n_tree_nodes, dtype=np.float64)
    nodes_eta = np.empty(n_tree_nodes, dtype=np.float64)
    nodes_phi = np.empty(n_tree_nodes, dtype=np.float64)
    nodes_mass = np.empty(n_tree_nodes, dtype=np.float64)
    nodes_d0 = np.empty(n_tree_nodes, dtype=np.float64)
    nodes_dz = np.empty(n_tree_nodes, dtype=np.float64)
    nodes_pt[:n_initial] = pt
    nodes_eta[:n_initial] = eta
    nodes_phi[:n_initial] = phi
    nodes_mass[:n_initial] = mass
    nodes_d0[:n_initial] = d0
    nodes_dz[:n_initial] = dz
    
    # New Features of constituents
    nodes_Delta = np.zeros(n_tree_nodes, dtype=np.float64)
    nodes_z = np.ones(n_tree_nodes, dtype=np.float64)
    nodes_psi = np.full(n_tree_nodes, np.pi/4, dtype=np.float64)
    nodes_kt = np.zeros(n_tree_nodes, dtype=np.float64)

    # Prepare index map to correctly build the Lund tree
    index_list = [i for i in range(constituents.shape[1])]
    next_node_index = n_initial

    # Start the actual CA clustering
    while constituents.shape[1] > 1:

        dij = get_dij(constituents)

        # Get pair with minimum distance
        i =  np.argmin(dij) // dij.shape[0]
        j =  np.argmin(dij) % dij.shape[0]

        # Get index of new node in the tree
        k = next_node_index
        next_node_index += 1

        # Cluster the pair together
        new_pt, new_eta, new_phi, new_mass = cluster_constituents(constituents, i, j)

        new_d0 = (nodes_d0[index_list[i]] + nodes_d0[index_list[j]]) / 2
        new_dz = (nodes_dz[index_list[i]] + nodes_dz[index_list[j]]) / 2

        nodes_d0[k] = new_d0
        nodes_dz[k] = new_dz
        nodes_pt[k] = new_pt
        nodes_eta[k] = new_eta
        nodes_phi[k] = new_phi
        nodes_mass[k] = new_mass
        
        # Order indices by constituent pt
        if constituents[0,i] < constituents[0,j]:
            i_lo, i_hi = (i, j)
        else:
            i_lo, i_hi = (j, i)
        
        # New Features for clustered particle
        rapidity_i = get_rapidity(constituents[0,i], constituents[1,i], constituents[3,i])
        rapidity_j = get_rapidity(constituents[0,j], constituents[1,j], constituents[3,j])
        Delta = get_delta(
            rapidity_i,
            rapidity_j,
            constituents[2,i],
            constituents[2,j]
        )
        if np.isnan(Delta):
            print("pt: ", constituents[0,i])
            print("eta: ", constituents[1,i])
            print("phi: ", constituents[2,i])
            print("mass: ", constituents[3,i])
            print("y_1", get_rapidity(constituents[0,i], constituents[1,i], constituents[3,i]))
            print("y_2", get_rapidity(constituents[0,j], constituents[1,j], constituents[3,j]))
            print(constituents[0,j], constituents[1,j], constituents[3,j])
            print("Delta: ", Delta)
            raise ValueError("NaN in Delta")
        
        z = get_z(constituents[0,i], constituents[0,j])
        if i_lo == i:
            psi = get_psi(rapidity_i, rapidity_j, constituents[2,i], constituents[2,j])
        else:
            psi = get_psi(rapidity_j, rapidity_i, constituents[2,j], constituents[2,i])
        kt = get_kt(constituents[0,i], constituents[0,j], Delta)
        
        nodes_Delta[k] = Delta
        nodes_z[k] = z
        nodes_psi[k] = psi
        nodes_kt[k] = kt

        if save_fractions:
            # Energy fractions of the current pseudojets
            energy_i = get_energy(constituents[0,i], constituents[1,i], constituents[3,i])
            energy_j = get_energy(constituents[0,j], constituents[1,j], constituents[3,j])
            energy_new = get_energy(new_pt, new_eta, new_mass)
            e_fraction[k] = (energy_i*e_fraction[index_list[i]] + energy_j*e_fraction[index_list[j]]) / energy_new
            mu_fraction[k] = (energy_i*mu_fraction[index_list[i]] + energy_j*mu_fraction[index_list[j]]) / energy_new
            g_fraction[k] = (energy_i*g_fraction[index_list[i]] + energy_j*g_fraction[index_list[j]]) / energy_new
            h_fraction[k] = (energy_i*h_fraction[index_list[i]] + energy_j*h_fraction[index_list[j]]) / energy_new
        
        # Add connections to the tree
        # This construction ensures that each clustered step is
        # two-way connected to its mother constituents
        edges[0, edge_offset:edge_offset + 4] = (
            index_list[i], index_list[j], k, k
        )
        edges[1, edge_offset:edge_offset + 4] = (
            k, k, index_list[i], index_list[j]
        )
        edge_offset += 4

        # Update the index map
        index_list = [idx for l, idx in enumerate(index_list) if l not in (i,j)] + [k]

        # Replace clustered constituents by the clustered particle
        
        pt = np.hstack(( 
            pt[:min(i,j)], 
            pt[min(i,j)+1:max(i,j)], 
            pt[max(i,j)+1:], 
            np.array([new_pt])
        ))
        eta = np.hstack(( 
            eta[:min(i,j)], 
            eta[min(i,j)+1:max(i,j)], 
            eta[max(i,j)+1:], 
            np.array([new_eta])
        ))
        phi = np.hstack(( 
            phi[:min(i,j)], 
            phi[min(i,j)+1:max(i,j)], 
            phi[max(i,j)+1:], 
            np.array([new_phi])
        ))
        mass = np.hstack(( 
            mass[:min(i,j)], 
            mass[min(i,j)+1:max(i,j)], 
            mass[max(i,j)+1:], 
            np.array([new_mass])
        ))

        constituents = np.stack((pt, eta, phi, mass), axis=0)

    # Somewhat cumbersome implementation needed for numba JIT compilation
    if not save_4vectors and not save_fractions:
        if n_lund_vars == 3:
            if use_displacement:
                nodes = np.vstack((
                    nodes_kt,
                    nodes_Delta,
                    nodes_z,
                    nodes_d0,   
                    nodes_dz,   
                ))
            else:
                nodes = np.vstack((
                    nodes_kt,
                    nodes_Delta,
                    nodes_z,
                ))

        elif n_lund_vars == 5:
            if use_displacement:  
                nodes = np.vstack((
                    nodes_kt,
                    nodes_Delta,
                    nodes_z,
                    nodes_psi,
                    nodes_mass,
                    nodes_d0,   
                    nodes_dz,   
                ))
            else:
                nodes = np.vstack((
                    nodes_kt,
                    nodes_Delta,
                    nodes_z,
                    nodes_psi,
                    nodes_mass,
                ))

    elif save_4vectors and not save_fractions:
        if n_lund_vars == 3:
            if use_displacement:
                nodes = np.vstack((
                    nodes_kt,
                    nodes_Delta,
                    nodes_z,
                    nodes_pt,
                    nodes_eta,
                    nodes_phi,
                    nodes_mass,
                    nodes_d0,   
                    nodes_dz,   
                ))
            else:
                nodes = np.vstack((
                    nodes_kt,
                    nodes_Delta,
                    nodes_z,
                    nodes_pt,
                    nodes_eta,
                    nodes_phi,
                    nodes_mass,
                ))
        elif n_lund_vars == 5:
            if use_displacement:
                nodes = np.vstack((
                    nodes_kt,
                    nodes_Delta,
                    nodes_z,
                    nodes_psi,
                    nodes_pt,
                    nodes_eta,
                    nodes_phi,
                    nodes_mass,
                    nodes_d0,   
                    nodes_dz,   
                ))
            else:
                nodes = np.vstack((
                    nodes_kt,
                    nodes_Delta,
                    nodes_z,
                    nodes_psi,
                    nodes_pt,
                    nodes_eta,
                    nodes_phi,
                    nodes_mass,
                ))

    elif save_fractions and not save_4vectors:
        if n_lund_vars == 3:
            if use_displacement:
                nodes = np.vstack((
                    nodes_kt,
                    nodes_Delta,
                    nodes_z,
                    e_fraction,
                    mu_fraction,
                    g_fraction,
                    h_fraction,
                    nodes_d0,   
                    nodes_dz,   
                ))
            else:
                nodes = np.vstack((
                    nodes_kt,
                    nodes_Delta,
                    nodes_z,
                    e_fraction,
                    mu_fraction,
                    g_fraction,
                    h_fraction,
                ))
        elif n_lund_vars == 5:
            if use_displacement:
                nodes = np.vstack((
                    nodes_kt,
                    nodes_Delta,
                    nodes_z,
                    nodes_psi,
                    nodes_mass,
                    e_fraction,
                    mu_fraction,
                    g_fraction,
                    h_fraction,
                    nodes_d0,   
                    nodes_dz,   
                ))
            else:
                nodes = np.vstack((
                    nodes_kt,
                    nodes_Delta,
                    nodes_z,
                    nodes_psi,
                    nodes_mass,
                    e_fraction,
                    mu_fraction,
                    g_fraction,
                    h_fraction,
                ))

    else:
        if n_lund_vars == 3:
            if use_displacement:
                nodes = np.vstack((
                    nodes_kt,
                    nodes_Delta,
                    nodes_z,
                    nodes_pt,
                    nodes_eta,
                    nodes_phi,
                    nodes_mass,
                    e_fraction,
                    mu_fraction,
                    g_fraction,
                    h_fraction,
                    nodes_d0,   
                    nodes_dz,   
                ))
            else:
                nodes = np.vstack((
                    nodes_kt,
                    nodes_Delta,
                    nodes_z,
                    nodes_pt,
                    nodes_eta,
                    nodes_phi,
                    nodes_mass,
                    e_fraction,
                    mu_fraction,
                    g_fraction,
                    h_fraction,
                ))
        elif n_lund_vars == 5:
            if use_displacement:
                nodes = np.vstack((
                    nodes_kt,
                    nodes_Delta,
                    nodes_z,
                    nodes_psi,
                    nodes_pt,
                    nodes_eta,
                    nodes_phi,
                    nodes_mass,
                    e_fraction,
                    mu_fraction,
                    g_fraction,
                    h_fraction,
                    nodes_d0,   
                    nodes_dz,   
                ))
            else:
                nodes = np.vstack((
                    nodes_kt,
                    nodes_Delta,
                    nodes_z,
                    nodes_psi,
                    nodes_pt,
                    nodes_eta,
                    nodes_phi,
                    nodes_mass,
                    e_fraction,
                    mu_fraction,
                    g_fraction,
                    h_fraction,
                ))

    return (nodes, edges)
