import os
import numpy as np
import uproot
import torch
import awkward as ak
from tqdm import tqdm
import logging
from torch_geometric.utils import negative_sampling
from utils.GraphBuilder import (
    GraphBuildConfig,
    build_pfnano_jet_graph,
    pfnano_branches,
)
import utils.awkward_array_utilities as akUtl

def __cast_unknown_type_branches(ak_array, field):

    type_text = str(ak.type(ak_array))

    if "unknown" in type_text:
        ak_array = akUtl.as_type(ak_array, np.float64)
        
    elif "?" in type_text:
        new_type_text = type_text.split("*")[-1].replace("?", "").replace(" ", "")
        ak_array = akUtl.as_type(ak_array, getattr(np, new_type_text))
        
    return ak_array




def __get_keys_from_file(file_):
    """Return keys of a ROOT file open with uproot4 without suffix ;1.

    Args:
        file_ (uproot.writing.writable.WritableDirectory)

    Returns:
        list[str]
    """

    return [key.split(";")[0] for key in file_.keys()]



def __get_collection_and_variable_names(field):
    """Get collection and variable names from field name.

    Args:
        field (str)

    Returns:
        tuple(str, str)
    """

    split = field.split("_")
    if len(split) > 2:
        return split[0], "_".join(split[1:])
    if len(split) == 2:
        return split[0], split[1]
    else:
        return split[0], None


def __get_jagged_collections(ak_array):
    """Get collections in the ak array.

    Args:
       ak_array (awkward.Array): ak array with fields

    Returns:
        list[str]
    """

    collections = []
    for field in ak_array.fields:
        collection_candidate, variable = __get_collection_and_variable_names(field)
        if variable is None or collection_candidate is None: continue
        if not "var" in str(ak.type(ak_array[field])): continue
        if len(ak_array[field]) == 1: continue
        if collection_candidate not in collections:
            collections.append(collection_candidate)

    return collections
        

def __make_nano_aod_event_tree(ak_array, sort_by_name=True):
    """Get object that can be written as a TTree to a ROOT file with uproot4.

    Args:
       ak_array (awkward.Array): ak array with fields

    Returns:
        list[str]
    """

    collections = __get_jagged_collections(ak_array)

    # Book dictionaries for arrays to zip together and that does not not have to be zipped
    branches_to_zip = {collection: {} for collection in collections}
    single_branches = {}
    
    branch_to_skip = []
    
    for field in ak_array.fields:
        for n_branch in ["n" + field.split("_")[0], "n" + field[0]]:
            if n_branch in ak_array.fields:
                branch_to_skip.append(n_branch)
    # Fill in those dictionaries
    for field in ak_array.fields:
        type_ = akUtl.get_type(ak_array[field])
        if field in branch_to_skip:
            continue
        if type_ in ["str", "string"]:
            continue
        if ak.count(ak.fill_none(ak_array[field], 0), axis=None) == 0:
            continue
        
        is_collection_variable = False
        
        collection_candidate, variable = __get_collection_and_variable_names(field)
        for collection in collections:
            if collection_candidate == collection:
                branches_to_zip[collection][variable] = __cast_unknown_type_branches(ak_array[field], field)
                is_collection_variable = True

        if not is_collection_variable:
            single_branches[field] = __cast_unknown_type_branches(ak_array[field], field)

    # Zip branches and add everything in a dict
    collections_branches = {}
    for collection in branches_to_zip.keys():

        counter = {}
        for branch_name, branch in branches_to_zip[collection].items():
            size = ak.count(branch, axis=None)
            if size not in counter.keys():
                counter[size] = 1
            else:
                counter[size] += 1
        
        if len(counter.keys()) != 1:
            highest_count_key = 0
            highest_count_value = 0
            for k, v in counter.items():
                if v > highest_count_value:
                    highest_count_key = k
                    highest_count_value = v
            items = dict(branches_to_zip[collection].items()).copy()
            for branch_name, branch in items.items():
                size = ak.count(branch, axis=None)
                if size != highest_count_key:
                    
                    single_branches[f"{collection}_{branch_name}"] = branches_to_zip[collection].pop(branch_name)
 
        try:
            collections_branches[collection] = ak.zip(branches_to_zip[collection])
        except:
            print(" ")

    branches = collections_branches
    if sort_by_name:
        single_branches_names = sorted(single_branches)
    else:
        single_branches_names = single_branches.keys()
    for branch_name in single_branches_names:
        if branch_name not in branches.keys():
            branches[branch_name] = single_branches[branch_name]

    return branches


def write_nano_aod_root_file(output_file_name, events=None, trees={}, mode="recreate"):
    """Write events opened with the BaseSchema to a ROOT file.
    
    Args:
        output_file_name (str)
        events (ak.Array)
        trees (dict[str, any]): Other trees with no collections
            Keys are tree name
            Values are tree content
        mode (str): "recreate" or "update"
    """


    with getattr(uproot, mode)(output_file_name) as output_file:
        if events is not None:
            output_file["Events"] = __make_nano_aod_event_tree(events)
        for tree_name, tree in trees.items():
            output_file[tree_name] = ak.Array(tree)

class ScoreProducer:

    """
    Produces new root files with GNN losses for each jet.
    Note:
    model = GNN model to use for scoring
    normalizer = normalizer for the normalization of the features (dictionary with keys "means" and "stds")
    """


    def __init__(
        self,
        root_file,
        delta_r_threshold=0.2,
        n_store_cands=None,
        max_events_to_process=None,
        use_relative_angles=True,
        save_n_constituents=False,
        save_event_number=False,
        n_lund_vars=3,
        save_4vectors_in_lund_tree=False,
        save_energy_fractions=True,
        kt_cut=None,
        weights=None,
        save_jet_idx=False,
        label:float=0.,
        verbose=False,
        input_format="PFNanoAOD",
        jet_collection="FatJet",
        use_displacement=False,
        model=None,
        normalizer=None,
        output_folder=None,
        alpha=1,
    ):

        self.root_file = root_file
        self.n_store_cands = n_store_cands
        self.delta_r_threshold = delta_r_threshold
        self.max_events_to_process = max_events_to_process
        self.use_relative_angles = use_relative_angles
        self.save_n_constituents = save_n_constituents
        self.save_event_number = save_event_number
        self.n_lund_vars = n_lund_vars
        self.save_4vectors_in_lund_tree = save_4vectors_in_lund_tree
        self.save_energy_fractions = save_energy_fractions
        self.kt_cut = kt_cut
        self.weights = weights
        self.save_jet_idx = save_jet_idx
        self.label = label
        self.verbose = verbose
        self.input_format = input_format
        self.jet_collection = jet_collection
        self.use_displacement = use_displacement
        self.model = model
        self.normalizer = normalizer
        self.output_folder = output_folder
        self.alpha = alpha

        if input_format != "PFNanoAOD":
            raise ValueError(
                "Only input_format='PFNanoAOD' is supported; "
                "TreeMaker support has been removed"
            )

        if self.output_folder:
            os.makedirs(self.output_folder, exist_ok=True)

        assert self.weights in ["xsec", None], "Only xsec and None are supported for the weights argument (more to be added later)"

        # Assert that no incompatible flags are used

        # If xsec weighs are requested but the number of events is hard capped, warn that an appoximation is used
        if self.weights == "xsec" and self.max_events_to_process:
            logging.warning("Using approximation for initial number of events with max number of events")
        self.model.eval()
        self.process()


    def _graph_build_config(self):
        return GraphBuildConfig(
            jet_collection=self.jet_collection,
            n_store_cands=self.n_store_cands,
            use_relative_angles=self.use_relative_angles,
            use_displacement=self.use_displacement,
            use_lund_decomp=True,
            n_lund_vars=self.n_lund_vars,
            save_4vectors_in_lund_tree=self.save_4vectors_in_lund_tree,
            save_energy_fractions=self.save_energy_fractions,
            kt_cut=self.kt_cut,
            label=self.label,
        )

    def process_event(
        self,
        i_start,
        i_stop,
        events,
        xsec,
        gen_events_before_selection,
        total_events,
        events_to_process,
        file_name,
    ):
        graphs = []
        config = self._graph_build_config()
        for event_index in tqdm(
            range(i_start, i_stop), disable=not self.verbose
        ):
            event = events[event_index]
            n_jets = int(event[f"n{self.jet_collection}"])
            for jet_index in range(n_jets):
                graph, n_constituents = build_pfnano_jet_graph(
                    event, jet_index, config
                )
                if graph is None:
                    logging.warning(
                        "Skipping event %s jet %s because it has no constituents",
                        event_index,
                        jet_index,
                    )
                    continue

                if self.weights == "xsec":
                    if not self.max_events_to_process:
                        weight = xsec / gen_events_before_selection
                    else:
                        weight = xsec / (
                            gen_events_before_selection
                            * events_to_process
                            / total_events
                        )
                    graph.w = torch.tensor([weight], dtype=torch.float)
                if self.save_n_constituents:
                    graph.n_constituents = torch.tensor(
                        [n_constituents], dtype=torch.long
                    )
                if self.save_event_number:
                    graph.event_number = torch.tensor(
                        [event_index], dtype=torch.long
                    )
                if self.save_jet_idx:
                    graph.jet_idx = torch.tensor(
                        [jet_index], dtype=torch.int
                    )

                graph.event_number = torch.tensor(
                    [event_index], dtype=torch.long
                )
                graph.jet_number = torch.tensor(
                    [jet_index], dtype=torch.int
                )
                graph.score = torch.tensor([0.0], dtype=torch.float)
                graph.file_name = file_name
                graphs.append(graph)
        return graphs
    def get_graphs(self, file_name):
        xsec = 1.0
        gen_events_before_selection = None
        if self.weights == "xsec":
            with uproot.open(file_name) as input_file:
                if "Metadata" not in input_file:
                    raise ValueError("No Metadata tree found in file")
                if "GenCrossSection" not in input_file["Metadata"].keys():
                    raise ValueError("No GenCrossSection found in Metadata")
                xsec = input_file["Metadata"]["GenCrossSection"].array(
                    entry_stop=1
                )[0]
                if "CutFlow" not in input_file:
                    raise ValueError("No CutFlow tree found in file")
                gen_events_before_selection = input_file["CutFlow"][
                    "Initial"
                ].array(entry_stop=1)[0]

        branches = pfnano_branches(self._graph_build_config())
        with uproot.open(f"{file_name}:Events") as tree:
            total_events = tree.num_entries
            events_to_process = (
                min(self.max_events_to_process, total_events)
                if self.max_events_to_process
                else total_events
            )
            events = tree.arrays(
                branches, entry_stop=events_to_process, library="ak"
            )
        if gen_events_before_selection is None:
            gen_events_before_selection = total_events
        return self.process_event(
            0,
            events_to_process,
            events,
            xsec,
            gen_events_before_selection,
            total_events,
            events_to_process,
            file_name,
        )
    def process(self):
        # Read data into huge `Data` list.
        graphs = []
        model = self.model
        model.eval()
        
        # Dictionary to hold scores
        scores = []
        graphs = self.get_graphs(self.root_file)

        # Compute scores for each graph
        for graph in tqdm(graphs, desc="Computing scores", disable=not self.verbose):
            x = graph.x
            edges = graph.edge_index
            negative_edge_index = negative_sampling(edges)
            if negative_edge_index.numel()==0:
                scores.append(float(-1.0))
            else:
                if self.normalizer is not None:
                    x = (x-self.normalizer["means"])/self.normalizer["stds"]
                
                # Calculate losses
                features_loss = model.features_loss(x, edges)
                edges_loss = model.edges_loss(
                    x,
                    edges,
                    negative_sampling(edges),
                )
                loss = edges_loss + self.alpha * features_loss

                scores.append(float(loss.detach()))

            

        # Read the original ROOT file
        with uproot.open(self.root_file) as root_file:
            events_tree = root_file["Events"]
            original_data = events_tree.arrays(library="ak")
            trees = {}
            if "Metadata" in root_file:
                trees["Metadata"] = root_file["Metadata"].arrays(library="ak")
            if "CutFlow" in root_file:
                trees["CutFlow"] = root_file["CutFlow"].arrays(library="ak")
            

            scores = ak.unflatten(scores, ak.num(root_file["Events"]["FatJet_pt"].array()))

            
        # Add new branch to the data
        original_data["FatJet_GNNScore"] = ak.Array(scores)

        # Write the updated data to a new ROOT file
        output_file = (os.path.join(self.output_folder, os.path.basename(self.root_file).replace(".root","_with_scores.root")) if self.output_folder else self.root_file.replace(".root","_with_scores.root"))

        print(f"Writing to {output_file}")
        write_nano_aod_root_file(
            output_file_name=output_file,
            events=original_data,
            trees=trees,
        )
