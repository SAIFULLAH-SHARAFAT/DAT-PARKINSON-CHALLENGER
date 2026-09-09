"""Run this entire cell after a fresh kernel start. Read-only, CPU-only diagnostic.

Set weights_file if automatic discovery is ambiguous. No prior cells, training
data, model downloads, or repository imports are used. NumPy/PyTorch must already
be installed for tensor checks. Only a sanitized JSON report is written.
"""

import os

DIAGNOSTIC_CONFIG = {
    "weights_file": os.environ.get("DAT_DINOV2_WEIGHTS", ""),
    "artifact_root": os.environ.get("DAT_ARTIFACT_ROOT", "/kaggle/working"),
    "output_file": os.environ.get("DAT_DIAGNOSTIC_OUTPUT", "/kaggle/working/phase66_weights_diagnostic_contract.json"),
}


def phase66_weight_diagnostic(config):
    import hashlib
    import json
    import os
    import platform
    import tempfile
    from pathlib import Path

    official_sha = "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"
    report = {
        "phase": "phase66_read_only_weights_diagnostic",
        "status": "diagnostic_started",
        "stage": "file_discovery",
        "python_version": platform.python_version(),
        "error_type": None,
        "training_performed": False,
        "challenge_data_read": False,
        "weights_modified": False,
        "paths_tensor_values_or_raw_logs_exported": False,
        "exact_phase66_model_constructor_and_strict_load_tested": False,
    }

    def sha_file(path):
        h = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    def expected_shapes():
        shapes = {
            "cls_token": (1, 1, 384), "mask_token": (1, 384),
            "pos_embed": (1, 1370, 384),
            "patch_embed.proj.weight": (384, 3, 14, 14),
            "patch_embed.proj.bias": (384,),
            "norm.weight": (384,), "norm.bias": (384,),
        }
        block = {
            "norm1.weight": (384,), "norm1.bias": (384,),
            "attn.qkv.weight": (1152, 384), "attn.qkv.bias": (1152,),
            "attn.proj.weight": (384, 384), "attn.proj.bias": (384,),
            "ls1.gamma": (384,), "norm2.weight": (384,),
            "norm2.bias": (384,), "mlp.fc1.weight": (1536, 384),
            "mlp.fc1.bias": (1536,), "mlp.fc2.weight": (384, 1536),
            "mlp.fc2.bias": (384,), "ls2.gamma": (384,),
        }
        for i in range(12):
            shapes.update({f"blocks.{i}.{k}": v for k, v in block.items()})
        return shapes

    protected_paths = set()
    try:
        names = ("dinov2_vits14_lvd142m.pth", "dinov2_vits14_pretrain.pth")
        if config["weights_file"]:
            candidates = [Path(config["weights_file"])]
        else:
            root = Path(config["artifact_root"])
            candidates = [root / name for name in names]
            if not any(p.is_file() for p in candidates):
                candidates += [p for name in names
                               for p in Path("/kaggle/input").glob("**/" + name)]
        candidates = list(dict.fromkeys(p.resolve() for p in candidates if p.is_file()))
        protected_paths.update(candidates)
        report["candidate_file_count"] = len(candidates)
        if not candidates:
            report["status"] = "missing_weights_set_weights_file"
        else:
            report["stage"] = "file_hash"
            digests = [sha_file(p) for p in candidates]
            report["distinct_file_digest_count"] = len(set(digests))
            if len(set(digests)) != 1:
                report["status"] = "ambiguous_weights_set_one_explicit_weights_file"
            else:
                report["file_sha256"] = digests[0]
                report["file_bytes"] = candidates[0].stat().st_size
                report["matches_public_reference_file_bytes"] = digests[0] == official_sha
                report["stage"] = "import_torch"
                import torch
                report["torch_version"] = str(torch.__version__)
                report["stage"] = "restricted_torch_load"
                obj = torch.load(candidates[0], map_location="cpu", weights_only=True)
                report["restricted_load_succeeded"] = True
                report["stage"] = "schema_and_finiteness"
                schema = expected_shapes()
                report["expected_tensor_count"] = len(schema)
                # Inspect a small set of known wrappers without changing the file.
                # Names found in the file are never included in the report.
                queue = [(obj, 0)]
                seen = set()
                mappings = []
                while queue:
                    node, depth = queue.pop(0)
                    if not isinstance(node, dict) or id(node) in seen or depth > 3:
                        continue
                    seen.add(id(node))
                    if node and all(isinstance(k, str) and torch.is_tensor(v)
                                    for k, v in node.items()):
                        mappings.append((node, depth))
                    for wrapper in ("state_dict", "model", "teacher", "student", "backbone"):
                        if wrapper in node:
                            queue.append((node[wrapper], depth + 1))
                observations = []
                for mapping, depth in mappings:
                    variants = [(mapping, 0)]
                    for current, prefix_count in variants:
                        keys = set(current)
                        shared = keys & set(schema)
                        missing = len(set(schema) - keys)
                        extra = len(keys - set(schema))
                        wrong = sum(tuple(current[k].shape) != schema[k] for k in shared)
                        compatible = missing == extra == wrong == 0
                        finite = None
                        if compatible:
                            finite = all(bool(torch.isfinite(v).all()) for v in current.values())
                        observations.append({
                            "wrapper_depth": depth, "prefix_removals_needed": prefix_count,
                            "tensor_count": len(current), "missing_key_count": missing,
                            "unexpected_key_count": extra, "shape_mismatch_count": wrong,
                            "schema_compatible": compatible, "all_finite": finite,
                        })
                        if prefix_count < 3:
                            for prefix in ("module.", "backbone.", "model.", "teacher.", "student."):
                                if all(k.startswith(prefix) for k in current):
                                    variants.append(({k[len(prefix):]: v for k, v in current.items()},
                                                     prefix_count + 1))
                report["tensor_mapping_count"] = len(mappings)
                report["schema_observations"] = observations
                compatible = [o for o in observations if o["schema_compatible"] and o["all_finite"]]
                raw = any(o["wrapper_depth"] == o["prefix_removals_needed"] == 0 for o in compatible)
                if raw and report["matches_public_reference_file_bytes"]:
                    report["status"] = "official_raw_weights_verified_in_this_runtime"
                elif raw:
                    report["status"] = "raw_schema_compatible_original_pretraining_provenance_unverified"
                elif compatible:
                    report["status"] = "compatible_schema_inside_wrapper_original_pretraining_provenance_unverified"
                else:
                    report["status"] = "no_complete_finite_nonregister_vits14_schema_found"
                report["stage"] = "complete"
    except Exception as exc:
        report["status"] = "diagnostic_runtime_stop"
        report["error_type"] = type(exc).__name__

    # Hash convention: canonical JSON before adding contract_sha256.
    report["contract_sha256"] = hashlib.sha256(
        json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    text = json.dumps(report, indent=2, allow_nan=False)
    try:
        output = Path(config["output_file"])
        if output.resolve() in protected_paths:
            raise ValueError("output_must_not_replace_input_weights")
        output.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix="phase66_diag_", dir=output.parent)
        try:
            with os.fdopen(fd, "w") as stream:
                stream.write(text + "\n")
            os.replace(temporary, output)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    except Exception:
        # The complete self-hashed report remains available in notebook output.
        print("Sanitized diagnostic file could not be saved; retain the block below.")
    print("BEGIN SANITIZED_PHASE66_WEIGHTS_DIAGNOSTIC")
    print(text)
    print("END SANITIZED_PHASE66_WEIGHTS_DIAGNOSTIC")
    return report


if __name__ == "__main__":
    phase66_weight_diagnostic(DIAGNOSTIC_CONFIG)
