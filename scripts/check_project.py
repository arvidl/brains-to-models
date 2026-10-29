"""Check the publication contents, paper references and saved notebooks."""

import ast
import json
import math
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]


def publication_files():
    if (ROOT / ".git").exists():
        names = subprocess.check_output(
            ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
            cwd=ROOT,
        ).decode().split("\0")
        return [ROOT / name for name in set(names) if name]
    # Also support source archives, which have no Git metadata.
    ignored = {".venv", ".build", "__pycache__", ".ipynb_checkpoints", ".mplconfig"}
    return [p for p in ROOT.rglob("*") if p.is_file()
            and not ignored.intersection(p.relative_to(ROOT).parts)]


def check_local_links(path, text):
    for target in re.findall(r"!?\[[^\]]*\]\(([^\s)]+)\)", text):
        parsed = urlsplit(target)
        if parsed.scheme or parsed.netloc or not parsed.path:
            continue
        linked = path.parent / unquote(parsed.path)
        assert linked.exists(), f"{path.relative_to(ROOT)}: missing link {target}"


def main():
    files = publication_files()
    excluded_roots = {"slides", "sources", "refs", "MyStuff", ".cursor"}
    presentations = {".ppt", ".pptx", ".pptm", ".odp", ".key", ".pps", ".ppsx"}
    for path in files:
        rel = path.relative_to(ROOT)
        assert rel.parts[0] not in excluded_roots, f"Development material: {rel}"
        assert path.suffix.lower() not in {".zip", ".7z", ".tar", ".gz"}, f"Archive: {rel}"
        if path.suffix.lower() in presentations:
            assert path.parent == ROOT / "talk" and path.suffix.lower() == ".pptx", rel
            assert path.stat().st_size < 100 * 1024**2, f"PPTX exceeds GitHub limit: {rel}"
        if path.suffix == ".tex":
            assert not re.search(r"\\documentclass(?:\[[^]]*\])?\{beamer\}", path.read_text()), rel
        if path.suffix == ".md":
            check_local_links(path, path.read_text())
        if path.suffix == ".py":
            ast.parse(path.read_text(), filename=str(rel))

    manuscript = "\n".join(p.read_text() for p in (ROOT / "paper").glob("*.tex"))
    bib = (ROOT / "paper/references.bib").read_text()
    keys = re.findall(r"@\w+\{([^,]+),", bib)
    assert keys and len(keys) == len(set(keys)), "Empty or duplicate bibliography keys"
    cited = {key.strip() for group in re.findall(r"\\cite\w*\{([^}]+)\}", manuscript)
             for key in group.split(",")}
    assert cited <= set(keys), f"Missing bibliography keys: {cited - set(keys)}"
    for figure in re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", manuscript):
        assert (ROOT / "paper" / figure).is_file(), f"Missing figure: {figure}"
    for included in re.findall(r"\\(?:input|include)\{([^}]+)\}", manuscript):
        assert (ROOT / "paper" / included).with_suffix(".tex").is_file(), included

    values = json.loads((ROOT / "paper/figures/computed_values.json").read_text())
    for prevalence, expected in [("0.01", 1 / 12), ("0.1", .5), ("0.5", .9)]:
        assert math.isclose(values["ppv"][prevalence], expected, rel_tol=1e-12)
    energies = values["hopfield_energy"]
    assert all(b <= a + 1e-10 for a, b in zip(energies, energies[1:]))
    assert values["hopfield_retrieves_pattern"]
    maps = json.loads((ROOT / "paper/figures/attention_maps_values.json").read_text())
    assert maps["gradient_check_max_abs_error"] < 1e-7
    assert all(math.isclose(sum(row), 1, abs_tol=1e-12) for row in maps["attention"])

    notebooks = sorted((ROOT / "code/notebooks").glob("*.ipynb"))
    assert notebooks, "No notebooks found"
    for path in notebooks:
        notebook = json.loads(path.read_text())
        assert notebook["nbformat"] == 4, path.name
        ids = [cell["id"] for cell in notebook["cells"]]
        assert len(ids) == len(set(ids)), f"Duplicate cell IDs: {path.name}"
        for cell in notebook["cells"]:
            assert cell["cell_type"] in {"code", "markdown", "raw"}, path.name
            assert all(ord(char) >= 32 or char in "\n\t"
                       for char in "".join(cell["source"])), (path.name, cell["id"])
            if cell["cell_type"] == "markdown":
                check_local_links(path, "".join(cell["source"]))
            assert not any(output.get("output_type") == "error"
                           for output in cell.get("outputs", [])), path.name
        if path.name == "13_verified_growth_model.ipynb":
            previews = sum("image/png" in output.get("data", {})
                           for cell in notebook["cells"]
                           for output in cell.get("outputs", []))
            assert previews >= 2, "Verification notebook is missing its saved plots"

    from brainsmodels.growth_verification import REFERENCE_COMMIT, REFERENCE_MODEL_SHA256
    resources = json.loads((ROOT / "talk/resources.json").read_text())
    reference = resources["references"]["brainweb_fisher_kpp"]
    assert reference["commit"] == REFERENCE_COMMIT
    assert reference["model_sha256"] == REFERENCE_MODEL_SHA256
    guide = (ROOT / "talk/resources.md").read_text()
    mapped = []
    for first, last in re.findall(r"^\| (\d+)(?:[–-](\d+))? \|", guide, re.MULTILINE):
        mapped.extend(range(int(first), int(last or first) + 1))
    assert sorted(mapped) == list(range(1, resources["deck"]["slide_count"] + 1)), \
        "Resource guide must cover each slide exactly once"
    for item in resources["references"].values():
        assert re.fullmatch(r"[0-9a-f]{40}", item["commit"])
        assert item["commit"] in guide, "Guide and resource versions differ"
    assert resources["deck"]["slide_count"] == 25
    assert re.fullmatch(r"[0-9a-f]{64}", resources["deck"]["sha256"])

    print(f"PASS: publication contents; {len(keys)} bibliography entries; paper figures; "
          f"local links; {len(notebooks)} notebooks; numerical references; Python syntax.")


if __name__ == "__main__":
    main()
