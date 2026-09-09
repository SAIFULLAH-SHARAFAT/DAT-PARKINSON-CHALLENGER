"""
Cell 165A-R2 — Phase56 provenance, reachability, and private replay audit

Run after Cell 165A-R. Phase56 artifacts are read-only. The cell never reads
challenge data unless the optional private replay bundle below is explicitly
provided in the live Kaggle kernel. It never serializes paths, UIDs, reference
probabilities, runtime probabilities, source text, or subprocess logs.

Optional private replay (all values remain in memory):

PHASE65AR2_REPLAY_PRIVATE = {
    # 1-D sequence of PRIVATE development NIfTI paths. Use a diverse, fixed
    # rehearsal subset; labels are neither needed nor permitted here.
    "nifti_paths": [...],
    # Development-path Phase56 probabilities in exactly the same order.
    "reference_probability": np.asarray([...], dtype=np.float64),
    # Optional; defaults match the competition schema.
    "uid_column": "uid",
    "prediction_column": "is_pathologic",
    "minimum_cases": 8,
    # Optional expected aggregate branch counts and preprocessing booleans from
    # the development replay. They are checked only when supplied.
    "reference_branch_counts": {"known": 6, "unknown": 2},
    "reference_preprocessing_checks": {"canonical_ras": True, ...},
}

The submitted archive is executed twice on the same rehearsal cases:
  1. instrumented, offline-blocked run for path/network observations;
  2. clean offline-environment run for exact output comparison.

Both runs execute the extracted final ZIP without editing its source. The cell
temporarily creates /code_execution only when that path does not already exist,
then removes only the symlink it created. If /code_execution already exists,
dynamic replay stops cleanly to avoid overwriting user or system data.
"""

from __future__ import annotations

import ast
import csv
import difflib
import hashlib
import json
import math
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import tokenize
import zipfile
from collections import Counter, defaultdict, deque
from io import BytesIO
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

import numpy as np


PHASE65AR2_STARTED = time.time()
PHASE65AR2_ROOT = Path(globals().get("PHASE65AR2_ROOT_OVERRIDE_PRIVATE", "/kaggle/working"))
PHASE65AR2_OFFICIAL_ROOT = Path(
    globals().get("PHASE65AR2_OFFICIAL_ROOT_OVERRIDE_PRIVATE", "/code_execution")
)
PHASE65AR2_CONTRACT_PATH = PHASE65AR2_ROOT / "phase65ar2_phase56_provenance_reachability_replay_contract.json"

PHASE65AR2_PATHS = {
    "submission_zip": PHASE65AR2_ROOT / "phase56_submission.zip",
    "source_handoff_zip": PHASE65AR2_ROOT / "phase56_runtime_source_handoff.zip",
    "staged_submission": PHASE65AR2_ROOT / "phase56_experimental_submission",
    "archive_build_contract": PHASE65AR2_ROOT / "phase56_archive_build_contract.json",
    "final_archive_contract": PHASE65AR2_ROOT / "phase56_final_archive_contract.json",
    "real_smoke_contract": PHASE65AR2_ROOT / "phase56_real_smoke_contract.json",
    "phase65a_contract": PHASE65AR2_ROOT / "phase65a_corrected_deployment_fallback_contract.json",
    "phase65ar_contract": PHASE65AR2_ROOT / "phase65ar_phase56_runtime_parity_contract.json",
}

PHASE65AR2_CONFIG = {
    "probability_atol": 1.0e-9,
    "probability_rtol": 0.0,
    "subprocess_timeout_seconds": 900,
    "maximum_structural_names": 40,
    "maximum_network_records": 40,
    "maximum_output_capture_bytes": 2_000_000,
    "require_zero_unreadable": True,
}


def phase65ar2_sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def phase65ar2_sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(block_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def phase65ar2_safe_json(path: Path) -> dict:
    result = {
        "exists": path.is_file(), "valid_object": False, "status": None,
        "sha256": None, "payload": None, "error_type": None,
    }
    if not result["exists"]:
        return result
    try:
        raw = path.read_bytes()
        result["sha256"] = phase65ar2_sha256_bytes(raw)
        obj = json.loads(raw.decode("utf-8"))
        result["valid_object"] = isinstance(obj, dict)
        if isinstance(obj, dict):
            result["payload"] = obj
            result["status"] = obj.get("status")
    except Exception as exc:
        result["error_type"] = type(exc).__name__
    return result


def phase65ar2_scalars(obj, key=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from phase65ar2_scalars(v, f"{key}.{k}" if key else str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from phase65ar2_scalars(v, f"{key}[{i}]")
    elif isinstance(obj, (str, int, float, bool)) or obj is None:
        yield key, obj


def phase65ar2_contract_digest_hits(payload: dict | None, digest: str) -> list[str]:
    if not isinstance(payload, dict) or not digest:
        return []
    hits = []
    for key, value in phase65ar2_scalars(payload):
        if "sha" in key.lower() and isinstance(value, str) and value.lower() == digest.lower():
            hits.append(key)
    return sorted(hits)


def phase65ar2_contract_mentions(payload: dict | None, token: str) -> bool:
    if not isinstance(payload, dict):
        return False
    token = token.lower()
    return any(token in str(v).lower() for _, v in phase65ar2_scalars(payload))


def phase65ar2_safe_member(name: str) -> bool:
    p = PurePosixPath(name)
    return bool(name) and "\\" not in name and not p.is_absolute() and ".." not in p.parts


def phase65ar2_zip_tree(path: Path, require_root_main: bool) -> dict:
    out = {
        "exists": path.is_file(), "valid": False, "sha256": None, "crc_ok": None,
        "unsafe_count": 0, "duplicate_count": 0, "symlink_count": 0,
        "file_count": 0, "root_main": False, "selected_main": None,
        "bytes": {}, "hashes": {}, "sources": {}, "error_type": None,
    }
    if not out["exists"]:
        return out
    try:
        out["sha256"] = phase65ar2_sha256_file(path)
        with zipfile.ZipFile(path) as zf:
            infos = zf.infolist()
            files = [i for i in infos if not i.is_dir()]
            names = [i.filename for i in infos]
            out["valid"] = True
            out["file_count"] = len(files)
            out["unsafe_count"] = sum(not phase65ar2_safe_member(n) for n in names)
            out["duplicate_count"] = len(names) - len(set(names))
            out["symlink_count"] = sum(stat.S_ISLNK((i.external_attr >> 16) & 0xFFFF) for i in infos)
            out["root_main"] = "main.py" in names
            mains = [i.filename for i in files if PurePosixPath(i.filename).name == "main.py"]
            if require_root_main:
                selected = "main.py" if out["root_main"] else None
            else:
                selected = min(mains, key=lambda n: (len(PurePosixPath(n).parts), n)) if mains else None
            out["selected_main"] = selected
            anchor = PurePosixPath(selected).parent if selected else None
            if anchor is not None:
                for info in files:
                    p = PurePosixPath(info.filename)
                    try:
                        rel = p.relative_to(anchor).as_posix()
                    except ValueError:
                        continue
                    data = zf.read(info)
                    out["bytes"][rel] = len(data)
                    out["hashes"][rel] = phase65ar2_sha256_bytes(data)
                    if rel.endswith(".py"):
                        out["sources"][rel] = data.decode("utf-8", errors="replace")
            out["crc_ok"] = zf.testzip() is None
    except Exception as exc:
        out["error_type"] = type(exc).__name__
    return out


def phase65ar2_dir_tree(path: Path) -> dict:
    out = {"exists": path.is_dir(), "file_count": 0, "selected_main": None,
           "bytes": {}, "hashes": {}, "sources": {}, "error_type": None}
    if not out["exists"]:
        return out
    try:
        files = [p for p in path.rglob("*") if p.is_file()]
        out["file_count"] = len(files)
        mains = [p for p in files if p.name == "main.py"]
        selected = min(mains, key=lambda p: (len(p.relative_to(path).parts), str(p))) if mains else None
        out["selected_main"] = str(selected.relative_to(path)) if selected else None
        if selected:
            anchor = selected.parent
            for p in files:
                try:
                    rel = p.relative_to(anchor).as_posix()
                except ValueError:
                    continue
                data = p.read_bytes()
                out["bytes"][rel] = len(data)
                out["hashes"][rel] = phase65ar2_sha256_bytes(data)
                if rel.endswith(".py"):
                    out["sources"][rel] = data.decode("utf-8", errors="replace")
    except Exception as exc:
        out["error_type"] = type(exc).__name__
    return out


def phase65ar2_ast_hash(text: str) -> str | None:
    try:
        tree = ast.parse(text)
        return phase65ar2_sha256_bytes(ast.dump(tree, annotate_fields=True, include_attributes=False).encode())
    except Exception:
        return None


def phase65ar2_token_hash(text: str) -> str | None:
    try:
        kept = []
        for tok in tokenize.tokenize(BytesIO(text.encode()).readline):
            if tok.type not in {tokenize.ENCODING, tokenize.COMMENT, tokenize.NL,
                                tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT,
                                tokenize.ENDMARKER}:
                kept.append((tok.type, tok.string))
        return phase65ar2_sha256_bytes(repr(kept).encode())
    except Exception:
        return None


def phase65ar2_def_hashes(text: str) -> dict[str, str]:
    try:
        tree = ast.parse(text)
    except Exception:
        return {}
    out = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            key = f"{type(node).__name__}:{node.name}"
            out[key] = phase65ar2_sha256_bytes(
                ast.dump(node, annotate_fields=True, include_attributes=False).encode()
            )
    return out


def phase65ar2_import_roots(text: str) -> set[str]:
    try:
        tree = ast.parse(text)
    except Exception:
        return set()
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(("." * node.level) + node.module)
    return roots


def phase65ar2_semantic_diff(left_text: str, right_text: str) -> dict:
    left_defs, right_defs = phase65ar2_def_hashes(left_text), phase65ar2_def_hashes(right_text)
    left_names, right_names = set(left_defs), set(right_defs)
    changed = sorted(k for k in left_names & right_names if left_defs[k] != right_defs[k])
    imports_l, imports_r = phase65ar2_import_roots(left_text), phase65ar2_import_roots(right_text)
    opcodes = Counter(x[0] for x in difflib.SequenceMatcher(
        a=left_text.splitlines(), b=right_text.splitlines(), autojunk=False
    ).get_opcodes())
    cap = PHASE65AR2_CONFIG["maximum_structural_names"]
    return {
        "raw_sha_match": phase65ar2_sha256_bytes(left_text.encode()) == phase65ar2_sha256_bytes(right_text.encode()),
        "comment_whitespace_insensitive_token_match": phase65ar2_token_hash(left_text) == phase65ar2_token_hash(right_text),
        "ast_match": phase65ar2_ast_hash(left_text) == phase65ar2_ast_hash(right_text),
        "added_definition_count": len(right_names - left_names),
        "removed_definition_count": len(left_names - right_names),
        "changed_definition_count": len(changed),
        "added_definitions": sorted(right_names - left_names)[:cap],
        "removed_definitions": sorted(left_names - right_names)[:cap],
        "changed_definitions": changed[:cap],
        "added_import_count": len(imports_r - imports_l),
        "removed_import_count": len(imports_l - imports_r),
        "added_imports": sorted(imports_r - imports_l)[:cap],
        "removed_imports": sorted(imports_l - imports_r)[:cap],
        "line_opcode_counts": dict(sorted(opcodes.items())),
        "source_lines_exported": False,
    }


def phase65ar2_module_name(rel: str) -> str:
    p = PurePosixPath(rel)
    parts = list(p.parts)
    if parts[-1] == "__init__.py":
        parts = parts[:-1]
    else:
        parts[-1] = p.stem
    return ".".join(parts)


def phase65ar2_import_targets(current: str, node, known: set[str]) -> set[str]:
    targets = set()
    if isinstance(node, ast.Import):
        candidates = [a.name for a in node.names]
    elif isinstance(node, ast.ImportFrom):
        base = node.module or ""
        if node.level:
            package = current.split(".")[:-1]
            keep = max(0, len(package) - node.level + 1)
            base = ".".join(package[:keep] + ([base] if base else []))
        candidates = [base] + [f"{base}.{a.name}" for a in node.names if base]
    else:
        candidates = []
    for c in candidates:
        probe = c
        while probe:
            if probe in known:
                targets.add(probe)
                break
            probe = probe.rsplit(".", 1)[0] if "." in probe else ""
    return targets


def phase65ar2_reachable_modules(source_map: dict[str, str]) -> tuple[set[str], dict[str, str]]:
    mod_to_rel = {phase65ar2_module_name(rel): rel for rel in source_map}
    known = set(mod_to_rel)
    entry = "main" if "main" in known else None
    if entry is None:
        return set(), mod_to_rel
    graph = defaultdict(set)
    for mod, rel in mod_to_rel.items():
        try:
            tree = ast.parse(source_map[rel])
        except Exception:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                graph[mod].update(phase65ar2_import_targets(mod, node, known))
    reached, queue = set(), deque([entry])
    while queue:
        cur = queue.popleft()
        if cur in reached:
            continue
        reached.add(cur)
        queue.extend(sorted(graph[cur] - reached))
    return reached, mod_to_rel


def phase65ar2_dotted_name(node) -> str | None:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def phase65ar2_domain(value: str) -> tuple[str | None, str | None]:
    try:
        parsed = urlparse(value)
        if parsed.scheme.lower() in {"http", "https"} and parsed.hostname:
            return parsed.scheme.lower(), parsed.hostname.lower()
    except Exception:
        pass
    return None, None


def phase65ar2_network_scan(source_map: dict[str, str], reachable_rels: set[str]) -> dict:
    network_modules = {"requests", "urllib", "httpx", "socket", "wget", "gdown", "aiohttp", "ftplib"}
    network_calls = {
        "requests.get", "requests.post", "requests.request", "httpx.get", "httpx.post",
        "urllib.request.urlopen", "urllib.request.urlretrieve", "socket.create_connection",
        "wget.download", "gdown.download", "torch.hub.load", "torch.hub.download_url_to_file",
    }
    records = []
    all_network_imports = 0
    reachable_network_imports = 0
    reachable_download_calls = 0
    for rel, text in sorted(source_map.items()):
        reachable = rel in reachable_rels
        try:
            tree = ast.parse(text, filename=rel)
        except Exception:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    if root in network_modules:
                        all_network_imports += 1
                        reachable_network_imports += int(reachable)
                        records.append({"file": rel, "line": int(node.lineno), "kind": "import",
                                        "name": alias.name, "reachable_from_main": reachable})
            elif isinstance(node, ast.ImportFrom) and node.module:
                root = node.module.split(".")[0]
                if root in network_modules:
                    all_network_imports += 1
                    reachable_network_imports += int(reachable)
                    records.append({"file": rel, "line": int(node.lineno), "kind": "import",
                                    "name": node.module, "reachable_from_main": reachable})
            elif isinstance(node, ast.Call):
                name = phase65ar2_dotted_name(node.func)
                dangerous = bool(name and (name in network_calls or any(name.endswith("." + x) for x in ("urlopen", "urlretrieve", "download"))))
                pretrained_true = bool(name and name.endswith("create_model") and any(
                    kw.arg == "pretrained" and isinstance(kw.value, ast.Constant) and kw.value.value is True
                    for kw in node.keywords
                ))
                if dangerous or pretrained_true:
                    reachable_download_calls += int(reachable)
                    records.append({"file": rel, "line": int(node.lineno), "kind": "call",
                                    "name": name, "reachable_from_main": reachable})
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                scheme, domain = phase65ar2_domain(node.value)
                if domain:
                    records.append({"file": rel, "line": int(getattr(node, "lineno", 0)),
                                    "kind": "url", "scheme": scheme, "domain": domain,
                                    "reachable_from_main": reachable})
    cap = PHASE65AR2_CONFIG["maximum_network_records"]
    return {
        "all_network_import_count": all_network_imports,
        "reachable_network_import_count": reachable_network_imports,
        "reachable_download_call_count": reachable_download_calls,
        "record_count": len(records),
        "records": records[:cap],
        "records_truncated": len(records) > cap,
        "static_reachability_is_not_runtime_invocation_proof": True,
        "source_text_exported": False,
    }


def phase65ar2_path_literals(source_map: dict[str, str], reachable_rels: set[str]) -> dict:
    official = {
        "niftis": ("/code_execution/data/niftis", "data/niftis"),
        "submission_format": ("/code_execution/data/submission_format.csv", "submission_format.csv"),
        "submission_output": ("/code_execution/submission.csv", "submission.csv"),
    }
    found = {k: False for k in official}
    candidates = 0
    for rel, text in source_map.items():
        if rel not in reachable_rels:
            continue
        try:
            tree = ast.parse(text)
        except Exception:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                value = node.value.replace("\\", "/")
                for key, tokens in official.items():
                    if any(token in value for token in tokens):
                        found[key] = True
                        candidates += 1
    return {"reachable_literal_or_component_presence": found,
            "matching_literal_count": candidates,
            "dynamic_observation_required_for_resolved_paths": True}


def phase65ar2_make_sitecustomize(directory: Path, trace_path: Path) -> None:
    # Trace only official-layout accesses and block outbound sockets. The trace
    # file contains categories, not private source paths or UIDs.
    code = r'''
import builtins, io, json, os, socket
_trace = os.environ.get("PHASE65AR2_TRACE_FILE")
_orig_open = builtins.open
_orig_io_open = io.open
_orig_listdir = os.listdir
_orig_scandir = os.scandir
_orig_connect = socket.socket.connect
_orig_create = socket.create_connection
_official = os.environ.get("PHASE65AR2_OFFICIAL_ROOT", "/code_execution").replace("\\", "/").rstrip("/")
def _emit(kind, value=""):
    try:
        if _trace:
            fd = os.open(_trace, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            os.write(fd, (json.dumps({"kind": kind, "value": value}) + "\n").encode())
            os.close(fd)
    except Exception:
        pass
def _category(path):
    try:
        p = os.fspath(path).replace("\\", "/")
    except Exception:
        return "other"
    if (_official + "/data/niftis") in p: return "niftis"
    if p.endswith(_official + "/data/submission_format.csv"): return "submission_format"
    if p.endswith(_official + "/submission.csv"): return "submission_output"
    return "other"
def _open(file, *a, **k):
    c = _category(file)
    if c != "other": _emit("path", c)
    return _orig_open(file, *a, **k)
def _io_open(file, *a, **k):
    c = _category(file)
    if c != "other": _emit("path", c)
    return _orig_io_open(file, *a, **k)
def _listdir(path="."):
    c = _category(path)
    if c != "other": _emit("path", c)
    return _orig_listdir(path)
def _scandir(path="."):
    c = _category(path)
    if c != "other": _emit("path", c)
    return _orig_scandir(path)
def _blocked_connect(self, address):
    host = str(address[0]) if isinstance(address, tuple) and address else "unknown"
    _emit("network_attempt", host)
    raise OSError("Phase65A-R2 offline replay blocked outbound network")
def _blocked_create(address, *a, **k):
    host = str(address[0]) if isinstance(address, tuple) and address else "unknown"
    _emit("network_attempt", host)
    raise OSError("Phase65A-R2 offline replay blocked outbound network")
builtins.open = _open
io.open = _io_open
os.listdir = _listdir
os.scandir = _scandir
socket.socket.connect = _blocked_connect
socket.create_connection = _blocked_create
'''
    (directory / "sitecustomize.py").write_text(code, encoding="utf-8")


def phase65ar2_read_trace(path: Path) -> dict:
    path_counts = Counter()
    network_count = 0
    domains = Counter()
    if path.is_file():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("kind") == "path":
                path_counts[str(rec.get("value"))] += 1
            elif rec.get("kind") == "network_attempt":
                network_count += 1
                value = str(rec.get("value", "unknown")).lower()
                # Do not export IPs or arbitrary values; domains only when DNS-like.
                if re.fullmatch(r"[a-z0-9.-]+", value) and not re.fullmatch(r"[0-9.]+", value):
                    domains[value] += 1
    return {
        "official_path_observed": {
            "niftis": path_counts["niftis"] > 0,
            "submission_format": path_counts["submission_format"] > 0,
            "submission_output": path_counts["submission_output"] > 0,
        },
        "official_path_access_counts": {k: int(path_counts[k]) for k in ("niftis", "submission_format", "submission_output")},
        "network_attempt_count": int(network_count),
        "network_attempt_domains": dict(sorted(domains.items())),
        "private_paths_exported": False,
    }


def phase65ar2_extract_safe(zip_path: Path, destination: Path) -> None:
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if not phase65ar2_safe_member(info.filename):
                raise ValueError("unsafe_zip_member")
            if stat.S_ISLNK((info.external_attr >> 16) & 0xFFFF):
                raise ValueError("zip_symlink_member")
        zf.extractall(destination)


def phase65ar2_load_probability_csv(path: Path, uid_column: str, prediction_column: str,
                                    expected_uids: list[str]) -> np.ndarray:
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != len(expected_uids):
        raise ValueError("output_row_count_mismatch")
    if not rows or uid_column not in rows[0] or prediction_column not in rows[0]:
        raise ValueError("output_schema_mismatch")
    mapping = {str(r[uid_column]): float(r[prediction_column]) for r in rows}
    if set(mapping) != set(expected_uids):
        raise ValueError("output_uid_set_mismatch")
    return np.asarray([mapping[u] for u in expected_uids], dtype=np.float64)


def phase65ar2_private_replay(submission_zip: Path) -> tuple[dict, dict | None]:
    bundle = globals().get("PHASE65AR2_REPLAY_PRIVATE")
    report = {
        "configured": isinstance(bundle, dict), "valid_configuration": False,
        "executed": False, "n": None, "instrumented_return_code": None,
        "clean_return_code": None, "instrumented_output_valid": False,
        "clean_output_valid": False, "instrumented_vs_clean_max_abs": None,
        "reference_vs_clean_max_abs": None, "reference_vs_clean_mean_abs": None,
        "instrumentation_invariant": None, "exact_reference_parity": None,
        "finite_and_in_range": None, "runtime_seconds_instrumented": None,
        "runtime_seconds_clean": None, "runtime_observation": None,
        "branch_counts_checked": False, "branch_counts_match": None,
        "preprocessing_checks_checked": False, "preprocessing_checks_all_true": None,
        "unreadable_count": None, "error_type": None, "error_stage": None,
        "case_level_values_exported": False, "paths_or_uids_exported": False,
        "subprocess_logs_exported": False,
    }
    if not isinstance(bundle, dict):
        return report, None
    created_official_link = False
    temp_parent = None
    try:
        paths = [Path(p) for p in bundle["nifti_paths"]]
        ref = np.asarray(bundle["reference_probability"], dtype=np.float64).reshape(-1)
        minimum = int(bundle.get("minimum_cases", 8))
        if len(paths) != ref.size or ref.size < minimum or not all(p.is_file() for p in paths):
            raise ValueError("invalid_paths_reference_or_minimum_cases")
        if not np.isfinite(ref).all() or not ((ref >= 0) & (ref <= 1)).all():
            raise ValueError("invalid_reference_probability")
        report["valid_configuration"] = True
        report["n"] = int(ref.size)
        uid_col = str(bundle.get("uid_column", "uid"))
        pred_col = str(bundle.get("prediction_column", "is_pathologic"))
        suffixes = [".nii.gz" if p.name.lower().endswith(".nii.gz") else ".nii" for p in paths]
        uids = [f"phase65ar2_{i:05d}" for i in range(len(paths))]

        temp_parent = Path(tempfile.mkdtemp(prefix="phase65ar2_", dir=str(PHASE65AR2_ROOT)))
        official_real = temp_parent / "official"
        data_dir = official_real / "data"
        nifti_dir = data_dir / "niftis"
        extract_dir = temp_parent / "submission"
        trace_dir = temp_parent / "trace_hook"
        trace_path = temp_parent / "trace.jsonl"
        nifti_dir.mkdir(parents=True)
        extract_dir.mkdir()
        trace_dir.mkdir()
        for uid, src, suffix in zip(uids, paths, suffixes):
            os.symlink(str(src.resolve()), str(nifti_dir / f"{uid}{suffix}"))
        with (data_dir / "submission_format.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[uid_col, pred_col])
            writer.writeheader()
            for uid in uids:
                writer.writerow({uid_col: uid, pred_col: 0.5})
        phase65ar2_extract_safe(submission_zip, extract_dir)
        if not (extract_dir / "main.py").is_file():
            raise ValueError("root_main_missing_after_extract")

        if PHASE65AR2_OFFICIAL_ROOT.exists() or PHASE65AR2_OFFICIAL_ROOT.is_symlink():
            raise FileExistsError("official_root_already_exists_refuse_overwrite")
        PHASE65AR2_OFFICIAL_ROOT.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(str(official_real), str(PHASE65AR2_OFFICIAL_ROOT))
        created_official_link = True

        phase65ar2_make_sitecustomize(trace_dir, trace_path)
        base_env = os.environ.copy()
        base_env.update({"PYTHONNOUSERSITE": "1", "TOKENIZERS_PARALLELISM": "false",
                         "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                         "TORCH_HOME": str(extract_dir / "torch_cache"),
                         "PHASE65AR2_OFFICIAL_ROOT": str(PHASE65AR2_OFFICIAL_ROOT)})

        def run_once(instrumented: bool):
            output = official_real / "submission.csv"
            if output.exists():
                output.unlink()
            env = base_env.copy()
            if instrumented:
                env["PYTHONPATH"] = str(trace_dir) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
                env["PHASE65AR2_TRACE_FILE"] = str(trace_path)
            else:
                env.pop("PHASE65AR2_TRACE_FILE", None)
            start = time.time()
            proc = subprocess.run(
                [sys.executable, "main.py"], cwd=str(extract_dir), env=env,
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                timeout=PHASE65AR2_CONFIG["subprocess_timeout_seconds"], check=False,
            )
            elapsed = time.time() - start
            # Logs remain private and are not placed in the report.
            captured = (proc.stdout + proc.stderr)[:PHASE65AR2_CONFIG["maximum_output_capture_bytes"]]
            return proc.returncode, elapsed, output, captured

        rc_i, sec_i, out_i, log_i_private = run_once(True)
        report["instrumented_return_code"] = int(rc_i)
        report["runtime_seconds_instrumented"] = float(sec_i)
        obs = phase65ar2_read_trace(trace_path)
        report["runtime_observation"] = obs
        if rc_i != 0 or not out_i.is_file():
            raise RuntimeError("instrumented_runtime_failed")
        pred_i = phase65ar2_load_probability_csv(out_i, uid_col, pred_col, uids)
        report["instrumented_output_valid"] = True

        rc_c, sec_c, out_c, log_c_private = run_once(False)
        report["clean_return_code"] = int(rc_c)
        report["runtime_seconds_clean"] = float(sec_c)
        if rc_c != 0 or not out_c.is_file():
            raise RuntimeError("clean_runtime_failed")
        pred_c = phase65ar2_load_probability_csv(out_c, uid_col, pred_col, uids)
        report["clean_output_valid"] = True
        report["executed"] = True

        finite_range = bool(np.isfinite(pred_i).all() and np.isfinite(pred_c).all()
                            and ((pred_i >= 0) & (pred_i <= 1)).all()
                            and ((pred_c >= 0) & (pred_c <= 1)).all())
        diff_ic = np.abs(pred_i - pred_c)
        diff_ref = np.abs(ref - pred_c)
        report["finite_and_in_range"] = finite_range
        report["instrumented_vs_clean_max_abs"] = float(diff_ic.max())
        report["reference_vs_clean_max_abs"] = float(diff_ref.max())
        report["reference_vs_clean_mean_abs"] = float(diff_ref.mean())
        report["instrumentation_invariant"] = bool(finite_range and np.allclose(
            pred_i, pred_c, atol=PHASE65AR2_CONFIG["probability_atol"],
            rtol=PHASE65AR2_CONFIG["probability_rtol"]))
        report["exact_reference_parity"] = bool(finite_range and np.allclose(
            ref, pred_c, atol=PHASE65AR2_CONFIG["probability_atol"],
            rtol=PHASE65AR2_CONFIG["probability_rtol"]))

        runtime_meta = bundle.get("runtime_metadata")
        if isinstance(runtime_meta, dict):
            if "unreadable_count" in runtime_meta:
                report["unreadable_count"] = int(runtime_meta["unreadable_count"])
            if isinstance(bundle.get("reference_branch_counts"), dict) and isinstance(runtime_meta.get("branch_counts"), dict):
                report["branch_counts_checked"] = True
                report["branch_counts_match"] = dict(bundle["reference_branch_counts"]) == dict(runtime_meta["branch_counts"])
            checks = runtime_meta.get("preprocessing_checks")
            if isinstance(checks, dict):
                report["preprocessing_checks_checked"] = True
                report["preprocessing_checks_all_true"] = bool(checks and all(bool(v) for v in checks.values()))
        return report, {"reference": ref, "instrumented": pred_i, "clean": pred_c,
                        "private_logs": (log_i_private, log_c_private)}
    except Exception as exc:
        report["error_type"] = type(exc).__name__
        report["error_stage"] = str(exc) if str(exc) in {
            "invalid_paths_reference_or_minimum_cases", "invalid_reference_probability",
            "root_main_missing_after_extract", "official_root_already_exists_refuse_overwrite",
            "instrumented_runtime_failed", "clean_runtime_failed", "output_row_count_mismatch",
            "output_schema_mismatch", "output_uid_set_mismatch", "unsafe_zip_member",
            "zip_symlink_member",
        } else "unexpected_private_replay_failure"
        return report, bundle
    finally:
        if created_official_link:
            try:
                if PHASE65AR2_OFFICIAL_ROOT.is_symlink() and Path(os.readlink(PHASE65AR2_OFFICIAL_ROOT)) == official_real:
                    PHASE65AR2_OFFICIAL_ROOT.unlink()
            except Exception:
                pass
        if temp_parent is not None:
            shutil.rmtree(temp_parent, ignore_errors=True)


def phase65ar2_jsonable(value):
    if isinstance(value, dict):
        return {str(k): phase65ar2_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [phase65ar2_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


# ------------------------------ audit ------------------------------

phase65ar2_contracts_private = {
    name: phase65ar2_safe_json(PHASE65AR2_PATHS[name])
    for name in ("archive_build_contract", "final_archive_contract", "real_smoke_contract",
                 "phase65a_contract", "phase65ar_contract")
}
phase65ar2_submission_private = phase65ar2_zip_tree(PHASE65AR2_PATHS["submission_zip"], True)
phase65ar2_handoff_private = phase65ar2_zip_tree(PHASE65AR2_PATHS["source_handoff_zip"], False)
phase65ar2_stage_private = phase65ar2_dir_tree(PHASE65AR2_PATHS["staged_submission"])

phase65ar2_submission_sha = phase65ar2_submission_private.get("sha256")
phase65ar2_contract_digest_hits_private = {
    name: phase65ar2_contract_digest_hits(rec.get("payload"), phase65ar2_submission_sha)
    for name, rec in phase65ar2_contracts_private.items()
}
phase65ar2_contract_summary = {
    name: {"exists": rec["exists"], "valid_json_object": rec["valid_object"],
           "status": rec["status"], "sha256": rec["sha256"],
           "submission_digest_hit_count": len(phase65ar2_contract_digest_hits_private[name])}
    for name, rec in phase65ar2_contracts_private.items()
}

submission_hashes = phase65ar2_submission_private.get("hashes", {})
stage_hashes = phase65ar2_stage_private.get("hashes", {})
handoff_hashes = phase65ar2_handoff_private.get("hashes", {})
stage_shared = set(submission_hashes) & set(stage_hashes)
stage_exact = bool(submission_hashes and set(submission_hashes).issubset(stage_hashes)
                   and all(submission_hashes[k] == stage_hashes[k] for k in submission_hashes))
handoff_shared_py = sorted(set(phase65ar2_submission_private.get("sources", {})) &
                           set(phase65ar2_handoff_private.get("sources", {})))
handoff_mismatch = [k for k in handoff_shared_py
                    if submission_hashes.get(k) != handoff_hashes.get(k)]

main_submission = phase65ar2_submission_private.get("sources", {}).get("main.py", "")
main_handoff = phase65ar2_handoff_private.get("sources", {}).get("main.py", "")
main_diff = phase65ar2_semantic_diff(main_handoff, main_submission) if main_submission and main_handoff else None

accepted_final = phase65ar2_contracts_private["final_archive_contract"]["status"] == "accepted"
accepted_smoke = phase65ar2_contracts_private["real_smoke_contract"]["status"] == "accepted"
final_records_digest = bool(phase65ar2_contract_digest_hits_private["final_archive_contract"])
build_records_digest = bool(phase65ar2_contract_digest_hits_private["archive_build_contract"])
smoke_links_final = (
    phase65ar2_contract_mentions(phase65ar2_contracts_private["real_smoke_contract"].get("payload"), "phase56")
    and accepted_smoke
)
authoritative_lineage = bool(
    phase65ar2_submission_private["valid"] and phase65ar2_submission_private["crc_ok"]
    and phase65ar2_submission_private["root_main"] and stage_exact
    and accepted_final and final_records_digest and accepted_smoke
)

reachable_mods, mod_to_rel = phase65ar2_reachable_modules(phase65ar2_submission_private.get("sources", {}))
reachable_rels = {mod_to_rel[m] for m in reachable_mods if m in mod_to_rel}
network_report = phase65ar2_network_scan(phase65ar2_submission_private.get("sources", {}), reachable_rels)
path_report = phase65ar2_path_literals(phase65ar2_submission_private.get("sources", {}), reachable_rels)
replay_report, replay_private = phase65ar2_private_replay(PHASE65AR2_PATHS["submission_zip"])

static_network_safe = network_report["reachable_download_call_count"] == 0
dynamic_network_safe = (
    replay_report["runtime_observation"] is not None
    and replay_report["runtime_observation"]["network_attempt_count"] == 0
)
runtime_paths_pass = bool(
    replay_report["runtime_observation"] is not None
    and all(replay_report["runtime_observation"]["official_path_observed"].values())
)
unreadable_pass = (
    replay_report["unreadable_count"] in (None, 0)
    if PHASE65AR2_CONFIG["require_zero_unreadable"] else True
)
replay_full_pass = bool(
    replay_report["executed"] and replay_report["instrumentation_invariant"]
    and replay_report["exact_reference_parity"] and dynamic_network_safe
    and runtime_paths_pass and unreadable_pass
    and replay_report["branch_counts_match"] is not False
    and replay_report["preprocessing_checks_all_true"] is not False
)

if not authoritative_lineage:
    status = "authoritative_final_archive_lineage_not_established_stop"
    recommendation = "Stop. Preserve Phase56; resolve the failed final-archive lineage criterion before replay or Phase65B."
elif network_report["reachable_download_call_count"] > 0 and not replay_report["executed"]:
    status = "archive_lineage_passed_reachable_network_call_requires_offline_replay"
    recommendation = "Run the private offline replay. A reachable network-capable call exists, but runtime invocation is not yet established."
elif not replay_report["configured"]:
    status = "archive_lineage_passed_private_replay_not_configured"
    recommendation = "Configure the in-kernel private replay bundle and rerun R2; do not rebuild Phase56 or advance Phase65B yet."
elif not replay_report["valid_configuration"]:
    status = "private_replay_configuration_invalid_stop"
    recommendation = "Correct only the private replay bundle. Phase56 remains unchanged."
elif not replay_report["executed"]:
    status = "private_replay_execution_failed_stop"
    recommendation = "Inspect the private subprocess logs in PHASE65AR2_REPLAY_STATE_PRIVATE; do not export them or rebuild Phase56 blindly."
elif not replay_full_pass:
    status = "private_runtime_or_development_parity_failed_stop"
    recommendation = "Stop. The final archive did not meet exact private replay, offline, path, or runtime metadata parity. Diagnose the failed criterion before Phase65B."
else:
    status = "accepted_phase56_provenance_reachability_and_exact_private_replay_ready_for_corrected_phase65b"
    recommendation = "Phase56 parity is established on the fixed private rehearsal. Proceed to corrected fold-local Phase65B; keep Phase56 as deployment anchor."

report = {
    "phase": "phase65ar2_phase56_provenance_reachability_and_private_replay_audit",
    "status": status,
    "artifact_provenance": {
        "submission_zip": {
            "exists": phase65ar2_submission_private["exists"],
            "sha256": phase65ar2_submission_sha,
            "file_count": phase65ar2_submission_private["file_count"],
            "valid_zip": phase65ar2_submission_private["valid"],
            "crc_ok": phase65ar2_submission_private["crc_ok"],
            "safe_paths": phase65ar2_submission_private["unsafe_count"] == 0,
            "root_main_present": phase65ar2_submission_private["root_main"],
        },
        "contracts": phase65ar2_contract_summary,
        "submission_matches_every_archived_stage_payload": stage_exact,
        "submission_stage_shared_file_count": len(stage_shared),
        "final_contract_accepted": accepted_final,
        "final_contract_records_exact_submission_digest": final_records_digest,
        "build_contract_records_exact_submission_digest": build_records_digest,
        "accepted_smoke_contract_present": accepted_smoke,
        "smoke_contract_links_phase56_lineage": smoke_links_final,
        "authoritative_final_archive_lineage_established": authoritative_lineage,
        "runtime_handoff_classification": (
            "non_authoritative_stale_or_pre_final_snapshot" if authoritative_lineage and handoff_mismatch
            else "source_identical_on_shared_python" if not handoff_mismatch and handoff_shared_py
            else "unresolved"
        ),
        "runtime_handoff_shared_python_count": len(handoff_shared_py),
        "runtime_handoff_mismatched_python_count": len(handoff_mismatch),
        "phase56_artifacts_modified": False,
    },
    "sanitized_main_structural_diff_handoff_to_final": main_diff,
    "module_reachability": {
        "archive_python_file_count": len(phase65ar2_submission_private.get("sources", {})),
        "reachable_from_main_file_count": len(reachable_rels),
        "unreachable_file_count": len(phase65ar2_submission_private.get("sources", {})) - len(reachable_rels),
        "reachability_is_conservative_import_graph": True,
    },
    "network_reachability": network_report,
    "runtime_path_resolution": {
        "static": path_report,
        "dynamic": replay_report["runtime_observation"],
        "all_required_paths_observed_during_replay": runtime_paths_pass if replay_report["executed"] else None,
    },
    "private_replay": replay_report,
    "gates": {
        "authoritative_lineage": authoritative_lineage,
        "no_reachable_network_capable_call_static": static_network_safe,
        "no_observed_network_attempt_dynamic": dynamic_network_safe if replay_report["executed"] else None,
        "runtime_paths_observed": runtime_paths_pass if replay_report["executed"] else None,
        "instrumentation_invariant": replay_report["instrumentation_invariant"],
        "exact_development_to_deployment_probability_parity": replay_report["exact_reference_parity"],
        "zero_unreadable_rehearsal_cases": unreadable_pass if replay_report["executed"] else None,
        "full_private_replay_passed": replay_full_pass,
    },
    "recommendation": recommendation,
    "interpretation_contract": {
        "stale_handoff_does_not_invalidate_digest_linked_final_archive": True,
        "static_import_or_url_is_not_runtime_network_invocation": True,
        "reachable_network_call_requires_dynamic_offline_evidence": True,
        "instrumented_and_clean_runs_use_unmodified_extracted_final_source": True,
        "public_leaderboard_used_for_gate": False,
        "phase65a_fallback_decision_changed": False,
        "phase56_remains_deployment_anchor": True,
    },
    "compliance": {
        "training_performed": False,
        "labels_read": False,
        "test_data_read": False,
        "test_set_adaptation": False,
        "private_development_scans_read_only_if_explicit_replay_configured": bool(replay_report["configured"]),
        "case_level_predictions_exported": False,
        "paths_or_uids_exported": False,
        "source_text_exported": False,
        "subprocess_logs_exported": False,
        "independent_test_case_inference_preserved": True,
    },
    "elapsed_seconds": float(time.time() - PHASE65AR2_STARTED),
}

# Private kernel state may include raw paths, predictions, and logs. Never save or print it.
PHASE65AR2_REPLAY_STATE_PRIVATE = {
    "report": report, "replay": replay_private, "configured_bundle": globals().get("PHASE65AR2_REPLAY_PRIVATE"),
    "submission_sources": phase65ar2_submission_private.get("sources", {}),
    "handoff_sources": phase65ar2_handoff_private.get("sources", {}),
}
PHASE65AR2_REPORT_PRIVATE = report

payload = phase65ar2_jsonable(report)
raw_without_hash = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
payload["contract_sha256"] = phase65ar2_sha256_bytes(raw_without_hash)
payload["contract_file"] = str(PHASE65AR2_CONTRACT_PATH)
PHASE65AR2_CONTRACT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

print("BEGIN SANITIZED_PHASE65AR2_PHASE56_PROVENANCE_REACHABILITY_REPLAY_AUDIT")
print(json.dumps(payload, indent=2))
print("END SANITIZED_PHASE65AR2_PHASE56_PROVENANCE_REACHABILITY_REPLAY_AUDIT")
