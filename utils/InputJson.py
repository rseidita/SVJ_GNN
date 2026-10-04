"""Input files described by JSON, one file per year and sample type.

Layout, written by ``make_input_json.py``::

    inputJson/<type>/<year>_<type>.json

with the content::

    {
      "year": "2018",
      "type": "background",
      "base": "root://host//store/user/.../nominal/",
      "datasets": {
        "QCD_HT1000to1500": ["root://host//store/.../part-0.root", ...],
        ...
      }
    }

Files are grouped by dataset because cross section weights need the number
of generated events of the dataset a file belongs to, summed over all of that
dataset's files.
"""

import json
import os
import re

import uproot


def normalize_input_path(path):
    """Return a path safe to pass to uproot.

    ``os.path.normpath`` turns ``root://host/...`` into ``root:/host/...``,
    which uproot then opens as a local file under the current directory.
    Relative local paths are left unchanged.
    """
    path = str(path).strip()
    if path.startswith("root:/") and not path.startswith("root://"):
        path = "root://" + path[6:]
    return path


def is_remote_path(path):
    return normalize_input_path(path).startswith("root://")


def part_index(file_path):
    """``part-N`` index from a skim filename, or the basename without ``.root``."""
    base = os.path.basename(normalize_input_path(file_path))
    match = re.match(r"part-(\d+)\.root$", base)
    if match:
        return match.group(1)
    return base.replace(".root", "")


def score_output_basename(role, file_path, sample_name=None, dataset=None):
    """Local score filename for one input ROOT file.

    Examples: ``data_part-0.root``, ``qcd_part-0.root``,
    ``signal_mMed-1000_mDark-20_rinv-0.3_part-0.root``.
    """
    part = part_index(file_path)
    if role == "data":
        prefix = "data"
    elif role == "background":
        prefix = "qcd"
    elif role == "signal":
        tag = None
        if sample_name:
            for stem in ("signal_2018_s-channel_", "signal_2017_s-channel_"):
                if sample_name.startswith(stem):
                    tag = sample_name[len(stem):]
                    break
        if tag is None and dataset:
            tag = dataset.replace("s-channel_", "")
        prefix = f"signal_{tag}" if tag else "signal"
    else:
        prefix = role
    return f"{prefix}_part-{part}.root"


def load_input_json(path):
    with open(path) as input_file:
        spec = json.load(input_file)
    if "datasets" not in spec or not isinstance(spec["datasets"], dict):
        raise ValueError(f"'{path}' has no 'datasets' mapping")
    if not any(spec["datasets"].values()):
        raise ValueError(f"'{path}' lists no input files")
    return spec


def write_input_json(path, year, sample_type, base, datasets):
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    spec = {
        "year": str(year),
        "type": sample_type,
        "base": base,
        "datasets": {
            name: sorted(files) for name, files in sorted(datasets.items())
        },
    }
    with open(path, "w") as output_file:
        json.dump(spec, output_file, indent=2)
        output_file.write("\n")
    return path


def dataset_groups(spec, datasets=None):
    """``{dataset: [files]}``, optionally restricted to named datasets."""
    groups = spec["datasets"]
    if datasets:
        missing = sorted(set(datasets) - set(groups))
        if missing:
            raise KeyError(
                f"Datasets {missing} are not in the input JSON; available: "
                f"{sorted(groups)}"
            )
        groups = {name: groups[name] for name in datasets}
    return {
        name: [normalize_input_path(path) for path in files]
        for name, files in groups.items()
    }


def flat_files(groups):
    """All files of *groups* in a stable dataset-then-file order."""
    return [
        normalize_input_path(file_path)
        for name in sorted(groups)
        for file_path in groups[name]
    ]


def dataset_of(file_path, groups):
    path = normalize_input_path(file_path)
    for name, files in groups.items():
        if path in files:
            return name
    raise KeyError(f"'{path}' belongs to no dataset in the input JSON")


def _path_match_keys(path):
    """Variants of *path* that should match the same input file."""
    path = normalize_input_path(path)
    keys = {path}
    if path.startswith("root://"):
        keys.add(os.path.normpath(path))
        keys.add(normalize_input_path(os.path.normpath(path)))
    return keys


def resolve_file_selection(all_paths, selected=None):
    """Match *selected* paths to *all_paths*, normalizing xrootd URLs."""
    canonical = [normalize_input_path(path) for path in all_paths]
    lookup = {}
    for path in canonical:
        for key in _path_match_keys(path):
            lookup[key] = path
    if selected is None:
        return canonical
    resolved = []
    for path in selected:
        match = None
        for key in _path_match_keys(path):
            if key in lookup:
                match = lookup[key]
                break
        if match is None:
            raise ValueError(
                f"Input file {path!r} is not listed for this sample; "
                f"expected one of {len(canonical)} configured path(s)"
            )
        resolved.append(match)
    return resolved


def shard_files(paths, files_per_job):
    """Split *paths* into contiguous shards of at most *files_per_job*."""
    if files_per_job < 1:
        raise ValueError("files_per_job must be at least 1")
    return [
        list(paths[start:start + files_per_job])
        for start in range(0, len(paths), files_per_job)
    ]


def sum_initial_events(files, verbose=False):
    """Sum ``CutFlow/Initial`` over *files*, the generated events of a dataset."""
    total = 0
    for index, file_path in enumerate(files):
        if verbose:
            print(f"  initial events {index + 1}/{len(files)}: {file_path}")
        with uproot.open(file_path) as input_file:
            if "CutFlow" not in input_file:
                raise ValueError(f"No CutFlow tree found in '{file_path}'")
            total += int(
                input_file["CutFlow"]["Initial"].array(entry_stop=1)[0]
            )
    return total
