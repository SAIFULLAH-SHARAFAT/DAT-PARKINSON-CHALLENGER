"""Fail when a script CI runs imports something the requirements do not declare.

This check exists because of a real failure: `scipy` was moved into
`requirements-archive.txt`, CI kept running an archived test that imports it, and
nothing installed it. The mistake survived local testing because the development
machine already had scipy from earlier work — so the only environment that could
have caught it was the clean one, which is to say CI, which is to say after the
badge had already gone red.

Static analysis, deliberately: it resolves imports by reading source, so it gives
the same answer on a machine that happens to have a package installed and on one
that does not.
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/ci.yml"

# Import name -> the distribution that provides it, where they differ.
DISTRIBUTION = {
    "sklearn": "scikit-learn",
    "cv2": "opencv-python",
    "PIL": "pillow",
    "yaml": "pyyaml",
}
# Installed by the workflow itself rather than by a requirements file.
WORKFLOW_PROVIDED = {"pip"}


def declared_distributions() -> set[str]:
    found = set()
    for name in ("requirements.txt", "requirements-archive.txt"):
        path = ROOT / name
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("-r"):
                continue
            found.add(re.split(r"[<>=!~;\[ ]", line)[0].strip().lower())
    return found


def scripts_ci_runs() -> list[Path]:
    """Every `python <path>.py` that appears in a `run:` block of the workflow."""
    text = WORKFLOW.read_text(encoding="utf-8")
    paths = []
    for match in re.finditer(r"python3?\s+(\S+\.py)\b", text):
        candidate = ROOT / match.group(1)
        if candidate.is_file():
            paths.append(candidate)
    return sorted(set(paths))


def _handles_missing_import(node: ast.Try) -> bool:
    for handler in node.handlers:
        caught = handler.type
        if caught is None:
            return True
        names = [caught] if not isinstance(caught, ast.Tuple) else list(caught.elts)
        for name in names:
            if isinstance(name, ast.Name) and name.id in {"ImportError", "ModuleNotFoundError"}:
                return True
    return False


def top_level_imports(path: Path) -> set[str]:
    """Imports the script genuinely needs, ignoring ones it knows may be absent.

    An import wrapped in `try: ... except ImportError:` is an optional
    dependency whose absence the code already handles, so requiring it to be
    declared would force the workflow to install something it does not need.
    Anything not so wrapped must be declared.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return set()

    optional = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Try) and _handles_missing_import(node):
            for inner in ast.walk(node):
                if isinstance(inner, ast.Import):
                    optional |= {a.name.split(".")[0] for a in inner.names}
                elif isinstance(inner, ast.ImportFrom) and inner.module and inner.level == 0:
                    optional.add(inner.module.split(".")[0])

    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module.split(".")[0])
    return names - optional


def local_module_paths(name: str, near: Path) -> list[Path]:
    """Where a bare `import x` could resolve inside this repository."""
    return [p for p in (near.parent / f"{name}.py", ROOT / f"{name}.py",
                        ROOT / name / "__init__.py") if p.is_file()]


def reachable(entry: Path, seen: set[Path]) -> set[str]:
    """Third-party imports of a script and of everything it pulls in.

    Follows local imports and `runpy.run_path` targets, because a notebook-cell
    export executed through runpy contributes its own imports to the run.
    """
    if entry in seen:
        return set()
    seen.add(entry)
    third_party = set()
    source = entry.read_text(encoding="utf-8")

    for name in top_level_imports(entry):
        if name in sys.stdlib_module_names or name == "__future__":
            continue
        local = local_module_paths(name, entry)
        if local:
            for path in local:
                third_party |= reachable(path, seen)
        else:
            third_party.add(name)

    for quoted in re.findall(r"""["'](\w+\.py)["']""", source):
        target = entry.parent / quoted
        if target.is_file():
            third_party |= reachable(target, seen)
    return third_party


def main() -> int:
    if not WORKFLOW.is_file():
        print("no workflow to check")
        return 0

    declared = declared_distributions()
    problems = []
    for script in scripts_ci_runs():
        rel = script.relative_to(ROOT).as_posix()
        for name in sorted(reachable(script, set())):
            if name in WORKFLOW_PROVIDED:
                continue
            distribution = DISTRIBUTION.get(name, name).lower()
            if distribution not in declared:
                problems.append(f"{rel} imports {name!r} -> {distribution!r} is not declared")

    if problems:
        print("DEPENDENCY CHECK FAILED", file=sys.stderr)
        for problem in problems:
            print(f"- {problem}", file=sys.stderr)
        print(
            "\nAdd the distribution to requirements.txt (or requirements-archive.txt "
            "if only the archive needs it) and install it in the workflow.",
            file=sys.stderr,
        )
        return 1

    print(f"DEPENDENCY CHECK PASSED ({len(scripts_ci_runs())} scripts, "
          f"{len(declared)} declared distributions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
