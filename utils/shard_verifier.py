#!/usr/bin/env python3
"""Utilities for verifying integrity and completeness of produced graph shards."""

import concurrent.futures
import io
import json
import os
import pickle
import re
import sys
import zipfile


class TorchStubUnpickler(pickle.Unpickler):
    """Unpickles metadata saved by PyTorch without importing torch or CUDA packages."""

    def find_class(self, module, name):
        if "torch" in module:
            return lambda *args, **kwargs: None
        return super().find_class(module, name)

    def persistent_load(self, pid):
        return None


def check_pt_file(path, deep=False):
    """Check if a PyTorch .pt file exists, is non-empty, and has a valid ZIP archive."""
    if not os.path.exists(path):
        return False, "File does not exist"
    try:
        size = os.path.getsize(path)
    except OSError as e:
        return False, f"Cannot stat file: {e}"
    if size == 0:
        return False, "File is empty (0 bytes)"

    try:
        with zipfile.ZipFile(path, "r") as zf:
            names = zf.namelist()
            if not names:
                return False, "Empty ZIP archive"
            if deep:
                crc_err = zf.testzip()
                if crc_err is not None:
                    return False, f"ZIP CRC check failed on {crc_err}"
            return True, "OK"
    except zipfile.BadZipFile as e:
        return False, f"BadZipFile: {e}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def get_meta_chunk_count(meta_path):
    """Extract n_chunks from a shard meta.pt file without requiring PyTorch."""
    try:
        with zipfile.ZipFile(meta_path, "r") as zf:
            for name in zf.namelist():
                if name.endswith("data.pkl"):
                    data = zf.read(name)
                    obj = TorchStubUnpickler(io.BytesIO(data)).load()
                    return obj.get("n_chunks", None), obj.get("n_graphs", None)
    except Exception:
        pass
    return None, None


def verify_single_shard(processed_dir, sample_name, shard_id, deep=False):
    """Verify completeness and integrity of a single shard.

    Returns:
        tuple: (shard_id, status, message, corrupted_files)
        where status is 'OK', 'MISSING', 'INCOMPLETE', or 'CORRUPTED'.
    """
    prefix = f"{sample_name}_shard{shard_id}"
    meta_path = os.path.join(processed_dir, f"{prefix}_meta.pt")

    if not os.path.exists(meta_path):
        # Check if any chunk file exists for this shard
        has_any_chunk = False
        try:
            for fname in os.listdir(processed_dir):
                if fname.startswith(f"{prefix}_chunk_"):
                    has_any_chunk = True
                    break
        except OSError:
            pass

        if not has_any_chunk:
            return shard_id, "MISSING", "Shard does not exist (no files found)", []
        return shard_id, "INCOMPLETE", "Missing meta.pt (job terminated prematurely)", []

    meta_ok, meta_err = check_pt_file(meta_path, deep=deep)
    if not meta_ok:
        return shard_id, "CORRUPTED", f"meta.pt corrupted: {meta_err}", [(meta_path, meta_err)]

    n_chunks, n_graphs = get_meta_chunk_count(meta_path)
    if n_chunks is None:
        return shard_id, "CORRUPTED", "Failed to read n_chunks from meta.pt", [(meta_path, "Unreadable meta")]

    corrupted_files = []
    missing_chunks = []
    for c in range(n_chunks):
        chunk_path = os.path.join(processed_dir, f"{prefix}_chunk_{c}.pt")
        chunk_ok, chunk_err = check_pt_file(chunk_path, deep=deep)
        if not chunk_ok:
            if "does not exist" in chunk_err:
                missing_chunks.append(c)
            else:
                corrupted_files.append((chunk_path, chunk_err))

    if corrupted_files:
        return shard_id, "CORRUPTED", f"{len(corrupted_files)} corrupted chunk file(s)", corrupted_files
    if missing_chunks:
        return shard_id, "INCOMPLETE", f"Missing chunk(s): {missing_chunks}", []

    return shard_id, "OK", f"{n_chunks} chunks verified ({n_graphs} graphs)", []


def count_expected_shards_from_json(json_path, files_per_job):
    """Calculate the expected number of shards directly from an input JSON."""
    with open(json_path, "r") as f:
        data = json.load(f)
    datasets = data.get("datasets", {})
    total_files = sum(len(flist) for flist in datasets.values())
    if files_per_job <= 0:
        raise ValueError("files_per_job must be greater than 0")
    return (total_files + files_per_job - 1) // files_per_job, total_files


def verify_sample_shards(processed_dir, sample_name, n_expected_shards, num_workers=32, deep=False):
    """Verify all expected shards for a given sample in parallel."""
    if not os.path.exists(processed_dir):
        return {
            "sample": sample_name,
            "processed_dir": processed_dir,
            "expected_shards": n_expected_shards,
            "dir_exists": False,
            "ok_shards": [],
            "missing_shards": list(range(n_expected_shards)),
            "incomplete_shards": [],
            "corrupted_shards": {},
        }

    tasks = [(processed_dir, sample_name, s, deep) for s in range(n_expected_shards)]
    ok_shards = []
    missing_shards = []
    incomplete_shards = []
    corrupted_shards = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(verify_single_shard, *task): task[2] for task in tasks}
        for fut in concurrent.futures.as_completed(futures):
            shard_id, status, reason, bad_files = fut.result()
            if status == "OK":
                ok_shards.append(shard_id)
            elif status == "MISSING":
                missing_shards.append(shard_id)
            elif status == "INCOMPLETE":
                incomplete_shards.append((shard_id, reason))
            elif status == "CORRUPTED":
                corrupted_shards[shard_id] = (reason, bad_files)

    ok_shards.sort()
    missing_shards.sort()
    incomplete_shards.sort(key=lambda x: x[0])

    return {
        "sample": sample_name,
        "processed_dir": processed_dir,
        "expected_shards": n_expected_shards,
        "dir_exists": True,
        "ok_shards": ok_shards,
        "missing_shards": missing_shards,
        "incomplete_shards": incomplete_shards,
        "corrupted_shards": corrupted_shards,
    }


def scan_directory_recursively(root_dir, num_workers=32, deep=False):
    """Scan any directory tree recursively and group verification by detected samples and shards."""
    pt_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.endswith(".pt"):
                pt_files.append(os.path.join(dirpath, fname))

    def _check(path):
        ok, err = check_pt_file(path, deep=deep)
        return path, ok, err

    corrupted_files = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        for path, ok, err in executor.map(_check, pt_files):
            if not ok:
                corrupted_files.append((path, err))

    # Group corrupted files by shard
    shard_pattern = re.compile(r"^(.*?)(?:_shard(\d+))")
    corrupted_by_sample = {}
    for path, err in corrupted_files:
        base = os.path.basename(path)
        match = shard_pattern.match(base)
        if match:
            s_name = match.group(1)
            shard_id = int(match.group(2)) if match.group(2) else None
        else:
            s_name = "unknown"
            shard_id = None
        corrupted_by_sample.setdefault(s_name, {}).setdefault(shard_id, []).append((path, err))

    return {
        "total_files": len(pt_files),
        "corrupted_files_count": len(corrupted_files),
        "corrupted_files": corrupted_files,
        "corrupted_by_sample": corrupted_by_sample,
    }


def generate_resubmit_jdl(template_path, failed_items, output_jdl_path, stage="produce"):
    """Write an HTCondor JDL containing only the failed shards in the Queue clause."""
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Template JDL not found: {template_path}")

    with open(template_path, "r") as f:
        content = f.read()

    queue_lines = [f"{stage}, {sample}, {shard}, {stage}, " for sample, shard in failed_items]
    queue_section = "Queue STAGE, SAMPLE, SHARD, LOG_SUBDIR, EXTRA_ARGS from (\n" + "\n".join(queue_lines) + "\n)\n"

    # Replace existing Queue clause
    match = re.search(r"Queue\s+.*?\s+from\s*\([\s\S]*?\)", content, flags=re.IGNORECASE)
    if match:
        new_content = content[:match.start()] + queue_section + content[match.end():]
    else:
        new_content = content + "\n" + queue_section

    os.makedirs(os.path.dirname(os.path.abspath(output_jdl_path)), exist_ok=True)
    with open(output_jdl_path, "w") as f:
        f.write(new_content)
    return output_jdl_path

