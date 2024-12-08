import numpy as np
import torch
from torch_geometric.transforms import BaseTransform
import numba as nb
import vector

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

    # Convert arrays into a binary tree representation for easier pruning
    # Find the top node of the tree
    top_node_index = np.max(edges[0])
    top_node = Node(nodes[top_node_index], top_node_index)

    # Start walking down the tree
    buffer = [top_node]
    visited_nodes = [top_node_index]
    while len(buffer) > 0:
        # Get the next node
        node = buffer.pop(0)
        visited_nodes.append(node.idx)

        # Find the children of the node
        node_children = [idx for idx in edges[1][np.where(edges[0] == node.idx)] if idx not in visited_nodes]

        # Check that the tree is binary as it should be
        if len(node_children) not in [0, 2]:
            raise ValueError("Lund Tree is not binary!")
        # Add the children to the tree
        if len(node_children) == 2:
            node.left = Node(nodes[node_children[0]], node_children[0])
            node.right = Node(nodes[node_children[1]], node_children[1])

            # Add the children to the buffer
            buffer.append(node.left)
            buffer.append(node.right)

    return top_node

def tree_to_tensor(top_node, shape):
        # Walk along the tree and convert it back into arrays
        nodes = np.zeros(shape)
        edges = [[], []]
        buffer = [top_node]
        top_node.idx = 0
        counter = 0
        nodes[0] = top_node.value

        while len(buffer) > 0:
            # Get the next node
            node = buffer.pop()
    
            # Add the children to the buffer
            if node.left is not None and node.right is not None:
                # Keep track of the global node index
                counter += 2
                node.left.idx = counter - 1
                node.right.idx = counter
                
                # Add the childrens features to the nodes array
                nodes[node.left.idx] = node.left.value
                nodes[node.right.idx] = node.right.value

                # Add the children to the buffer
                buffer.append(node.left)
                buffer.append(node.right)

                # Add the edges to the edges array
                edges[0].append(node.idx)
                edges[1].append(node.left.idx)
                edges[0].append(node.idx)
                edges[1].append(node.right.idx)
                edges[1].append(node.idx)
                edges[0].append(node.left.idx)
                edges[1].append(node.idx)
                edges[0].append(node.right.idx)
    
        edges = np.array(edges)
        nodes = np.array(nodes)
    
        return torch.tensor(nodes, dtype=torch.float), torch.tensor(edges, dtype=torch.long)

def prune_tree(top_node, cut_variable_idx, cut_value):

    # Walk along the tree and prune nodes that do not pass the cut
    buffer = [top_node]
    n_nodes = 0
    while len(buffer) > 0:
        # Get the next node
        node = buffer.pop(0)

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


@nb.njit
def get_rapidity(pt, eta, mass):
    return np.log((np.sqrt(mass**2 + (pt**2)*(np.cosh(eta)**2)) + pt*np.sinh(eta)) / np.sqrt(mass**2 + pt**2) + 1e-15)

@nb.njit
def get_delta(rapidity_a, rapidity_b, phi_a, phi_b):
    return np.sqrt((phi_a - phi_b)**2 + (rapidity_a - rapidity_b)**2)

@nb.njit
def get_z(pt_a, pt_b):
    return min(pt_a, pt_b) / (pt_a + pt_b)

@nb.njit
def get_psi(rapidity_low, rapidity_high, phi_low, phi_high):
    return np.arctan((rapidity_low - rapidity_high) / (phi_low - phi_high + 1e-15))

@nb.njit
def get_kt(pt_low, pt_high, delta):
    return min(pt_low, pt_high) * delta

@nb.njit
def get_energy(pt, eta, mass):
    return np.sqrt(mass**2 + pt**2)*np.cosh(eta)

@nb.njit
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
    matrix_y = np.repeat(y, len(y)).reshape((len(y), -1))
    matrix_phi = np.repeat(phi, len(phi)).reshape((len(phi), -1))
    dij = ( (matrix_phi - matrix_phi.T)**2 + (matrix_y - matrix_y.T)**2 ) / 0.64

    return dij

@nb.njit
def cluster_constituents(constituents:np.ndarray, index_a:int, index_b:int):
    constituent_a = vector.obj(pt=constituents[0, index_a], eta=constituents[1, index_a], phi=constituents[2, index_a], mass=constituents[3, index_a])
    constituent_b = vector.obj(pt=constituents[0, index_b], eta=constituents[1, index_b], phi=constituents[2, index_b], mass=constituents[3, index_b])

    return constituent_a + constituent_b


@nb.njit
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

    constituents = np.stack((pt, eta, phi, mass), axis=0)

    if save_fractions:
        e_fraction = fractions[0].astype(np.float64)
        mu_fraction = fractions[1].astype(np.float64)
        g_fraction = fractions[2].astype(np.float64)
        h_fraction = fractions[3].astype(np.float64)

    # Prepare sparse connection matrix for the Lund tree
    edges = np.zeros((2, 1), dtype=np.int64)

    pt = pt.astype(np.float64)
    eta = eta.astype(np.float64)
    phi = phi.astype(np.float64)
    mass = mass.astype(np.float64)

    # Features of constituents
    nodes_pt = pt.astype(np.float64).copy()
    nodes_eta = eta.astype(np.float64).copy()
    nodes_phi = phi.astype(np.float64).copy()
    nodes_mass = mass.astype(np.float64).copy()

    #Displacement Features of constituents
    nodes_d0 = d0.astype(np.float64).copy() 
    nodes_dz = dz.astype(np.float64).copy() 
    
    # New Features of constituents
    nodes_Delta = np.zeros(len(nodes_pt), dtype=np.float64)
    nodes_z = np.ones(len(nodes_pt), dtype=np.float64)
    nodes_psi = np.full(len(nodes_pt), np.pi/4, dtype=np.float64)
    nodes_kt = np.zeros(len(nodes_pt), dtype=np.float64)

    # Prepare index map to correctly build the Lund tree
    index_list = [i for i in range(constituents.shape[1])]

    first_clustering = True

    # Start the actual CA clustering
    while constituents.shape[1] > 1:

        dij = get_dij(constituents)

        # Avoiding the minimum distance to be the trivial 0 of each constituent with itself
        np.fill_diagonal(dij, np.inf)

        # Get pair with minimum distance
        i =  np.argmin(dij) // dij.shape[0]
        j =  np.argmin(dij) % dij.shape[0]

        # Get index of new node in the tree
        k = nodes_pt.shape[0]

        # Cluster the pair together
        new_constituent = cluster_constituents(constituents, i, j)

        new_d0 = (nodes_d0[i] + nodes_d0[j])/2 
        new_dz = (nodes_dz[i] + nodes_dz[j])/2 

        nodes_d0 = np.append(nodes_d0, new_d0) 
        nodes_dz = np.append(nodes_dz, new_dz) 

        nodes_pt = np.append(nodes_pt, new_constituent.pt)
        nodes_eta = np.append(nodes_eta, new_constituent.eta)
        nodes_phi = np.append(nodes_phi, new_constituent.phi)
        nodes_mass = np.append(nodes_mass, new_constituent.mass)
        
        # Order indices by constituent pt
        if constituents[0,i] < constituents[0,j]:
            i_lo, i_hi = (i, j)
        else:
            i_lo, i_hi = (j, i)
        
        # New Features for clustered particle
        Delta = get_delta(
            get_rapidity(constituents[0,i], constituents[1,i], constituents[3,i]),
            get_rapidity(constituents[0,j], constituents[1,j], constituents[3,j]),
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
        psi = get_psi(
            get_rapidity(constituents[0,i_lo], constituents[1,i_lo], constituents[3,i_lo]),
            get_rapidity(constituents[0,i_hi], constituents[1,i_hi], constituents[3,i_hi]),
            constituents[2,i_lo],
            constituents[2,i_hi]
        )
        kt = get_kt(constituents[0,i], constituents[0,j], Delta)
        
        nodes_Delta = np.append(nodes_Delta, Delta)
        nodes_z = np.append(nodes_z, z)
        nodes_psi = np.append(nodes_psi, psi)
        nodes_kt = np.append(nodes_kt, kt)

        if save_fractions:
            # Energy fractions of the current pseudojets
            energy_i = get_energy(constituents[0,i], constituents[1,i], constituents[3,i])
            energy_j = get_energy(constituents[0,j], constituents[1,j], constituents[3,j])
            energy_new = get_energy(new_constituent.pt, new_constituent.eta, new_constituent.mass)
            e_fraction = np.append(e_fraction, (energy_i*e_fraction[index_list[i]] + energy_j*e_fraction[index_list[j]]) / energy_new)
            mu_fraction = np.append(mu_fraction, (energy_i*mu_fraction[index_list[i]] + energy_j*mu_fraction[index_list[j]]) / energy_new)
            g_fraction = np.append(g_fraction, (energy_i*g_fraction[index_list[i]] + energy_j*g_fraction[index_list[j]]) / energy_new)
            h_fraction = np.append(h_fraction, (energy_i*h_fraction[index_list[i]] + energy_j*h_fraction[index_list[j]]) / energy_new)
        
        # Add connections to the tree
        # This construction ensures that each clustered step is
        # two-way connected to its mother constituents
        if first_clustering:
            edges = np.array([
                [index_list[i], index_list[j], k, k],
                [k, k, index_list[i], index_list[j]]
            ], dtype=np.int64
            )
            first_clustering = False
        else:
            new_edges = np.array([
                [index_list[i], index_list[j], k, k],
                [k, k, index_list[i], index_list[j]]
            ], dtype=np.int64
            )
            edges = np.hstack((edges, new_edges))

        # Update the index map
        index_list = [idx for l, idx in enumerate(index_list) if l not in (i,j)] + [k]

        # Replace clustered constituents by the clustered particle
        
        pt = np.hstack(( 
            pt[:min(i,j)], 
            pt[min(i,j)+1:max(i,j)], 
            pt[max(i,j)+1:], 
            np.array([new_constituent.pt])
        ))
        eta = np.hstack(( 
            eta[:min(i,j)], 
            eta[min(i,j)+1:max(i,j)], 
            eta[max(i,j)+1:], 
            np.array([new_constituent.eta])
        ))
        phi = np.hstack(( 
            phi[:min(i,j)], 
            phi[min(i,j)+1:max(i,j)], 
            phi[max(i,j)+1:], 
            np.array([new_constituent.phi])
        ))
        mass = np.hstack(( 
            mass[:min(i,j)], 
            mass[min(i,j)+1:max(i,j)], 
            mass[max(i,j)+1:], 
            np.array([new_constituent.mass])
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