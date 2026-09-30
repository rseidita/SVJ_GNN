import os


def read_file_list(path):
    """Return the ROOT file paths listed in a text file.

    One path per line. Blank lines and lines starting with '#' are ignored.
    Paths are kept verbatim, so local paths and 'root://' URLs both work.
    """
    with open(path) as input_file:
        paths = [
            line.strip()
            for line in input_file
            if line.strip() and not line.strip().startswith("#")
        ]
    if not paths:
        raise ValueError(f"File list '{path}' contains no file paths")
    return paths


def resolve_file_list(path, base_dir=None):
    """Read a file list and make relative local paths absolute.

    Relative paths are interpreted with respect to *base_dir*, defaulting to
    the directory holding the list, so a list can be moved without breaking.
    Remote URLs are returned unchanged.
    """
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(path))
    resolved = []
    for file_path in read_file_list(path):
        if "://" in file_path or os.path.isabs(file_path):
            resolved.append(file_path)
        else:
            resolved.append(os.path.normpath(os.path.join(base_dir, file_path)))
    return resolved


def shard_file_list(paths, files_per_job):
    """Split *paths* into contiguous shards of at most *files_per_job* entries."""
    if files_per_job < 1:
        raise ValueError("files_per_job must be at least 1")
    return [
        list(paths[start:start + files_per_job])
        for start in range(0, len(paths), files_per_job)
    ]


def write_file_list(paths, path):
    """Write one path per line, creating parent directories as needed."""
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    with open(path, "w") as output_file:
        for file_path in paths:
            output_file.write(f"{file_path}\n")
    return path
