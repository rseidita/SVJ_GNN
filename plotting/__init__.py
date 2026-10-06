"""Plotting package for SVJ_GNN."""

from plotting.plot_variables import VARIABLES, extract_variable, required_branches_for_vars
from plotting.plot_cuts import cut_branches, get_event_selection, post_hem_mask
from plotting.plot_style import setup_mplhep_style, BKG_ORDER, BKG_STYLE
from plotting.hist_reader import TH1Accumulator, run_histogram_generation, load_histograms_from_root, save_histograms_to_root
from plotting.plot_stack import plot_stacked_variable, plot_all_variables
from plotting.cutflow import run_cutflow

