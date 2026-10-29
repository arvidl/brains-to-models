"""Check the publication contents, paper references and saved notebooks."""

import ast
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote, urlsplit
import xml.etree.ElementTree as ET
import zipfile

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
        public_prefix = "https://github.com/arvidl/brains-to-models/blob/main/"
        if target.startswith(public_prefix):
            # Absolute repository links also work when a notebook opens in Colab.
            linked = ROOT / unquote(parsed.path.split("/blob/main/", 1)[1])
            assert linked.exists(), f"{path.relative_to(ROOT)}: missing link {target}"
            continue
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
        assert path.suffix.lower() not in {".zip", ".7z", ".tar", ".gz"} or (
            path.parent == ROOT / "code/src/brainsmodels/data/brainweb"
            and path.name in {"t1_icbm_normal_1mm_pn0_rf0.nii.gz",
                              "phantom_1.0mm_normal_crisp.nii.gz"}), f"Archive: {rel}"
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
    for notebook in re.findall(r"\\coursenotebook\{([^}]+)\}", manuscript):
        assert (ROOT / "code/notebooks" / notebook).is_file(), f"Missing paper notebook link: {notebook}"

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
        badge = ("https://colab.research.google.com/github/arvidl/brains-to-models/"
                 f"blob/main/code/notebooks/{path.name}")
        introduction = "".join(notebook["cells"][0]["source"])
        assert badge in introduction, f"Missing or incorrect Colab badge: {path.name}"
        markdown = "\n".join("".join(cell["source"]) for cell in notebook["cells"]
                             if cell["cell_type"] == "markdown")
        assert notebook["metadata"].get("colab", {}).get("name") == path.name, \
            f"Incorrect Colab title: {path.name}"
        for audience in ("For biomedical and imaging readers", "For computational readers"):
            assert audience in markdown, f"Missing reading path in {path.name}: {audience}"
        resource_lines = [line for line in markdown.splitlines()
                          if line.startswith("**Related resources:**")]
        expected_resources = (
            "**Related resources:** "
            "[Paper](https://github.com/arvidl/brains-to-models/blob/main/paper/main.pdf)"
            " · [Presentation guide]"
            "(https://github.com/arvidl/brains-to-models/blob/main/talk/resources.md)"
        )
        assert resource_lines == [expected_resources], \
            f"Missing or repeated course navigation: {path.name}"
        assert expected_resources in introduction, path.name
        editorial_history = re.search(
            r"August (?:background|perspective|manuscript)|September (?:draft|manuscript)"
            r"|Section numbers in the background readings|Section references follow"
            r"|(?:20|30)[-– ]minute (?:talk|presentation)", markdown,
            re.IGNORECASE,
        )
        assert editorial_history is None, \
            f"Editorial history in {path.name}: {editorial_history.group(0)}"
        setup = next(cell for cell in notebook["cells"] if cell["cell_type"] == "code")
        assert "setup" in setup["metadata"].get("tags", []), path.name
        setup_source = "".join(setup["source"])
        assert "https://github.com/arvidl/brains-to-models.git" in setup_source, path.name
        assert "brains-to-models-publication" not in setup_source, path.name
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
        if path.name.startswith(("14a_", "14b_", "14c_")):
            previews = sum("image/png" in output.get("data", {})
                           for cell in notebook["cells"] for output in cell.get("outputs", []))
            assert previews >= 1, f"Growth lesson is missing its saved figure: {path.name}"
            assert "ipywidgets>=8.1" in setup_source and "enable_custom_widget_manager" in setup_source
            assert any("interactive" in cell.get("metadata", {}).get("tags", [])
                       and "brainsmodels.growth_widgets" in "".join(cell["source"])
                       for cell in notebook["cells"]), f"Missing live growth controls: {path.name}"
        if path.name == "15_ppv_and_decisions.ipynb":
            assert "ipywidgets>=8.1" in setup_source and "enable_custom_widget_manager" in setup_source
            assert any("widget-check" in cell["metadata"].get("tags", [])
                       for cell in notebook["cells"]), "Missing decision-widget checks"
            for cell in notebook["cells"]:
                if cell["cell_type"] == "code":
                    ast.parse("".join(cell["source"]), filename=path.name)
            previews = sum("image/png" in output.get("data", {})
                           for cell in notebook["cells"] for output in cell.get("outputs", []))
            assert previews >= 2, "Decision notebook is missing its saved reference plots"

    from brainsmodels.brainweb_data import COMMIT, FILES, BUNDLED, verify_files
    verify_files(BUNDLED)
    manifest = json.loads((BUNDLED / "manifest.json").read_text())
    growth = json.loads((ROOT / "paper/figures/growth_extension_values.json").read_text())
    assert growth["data_commit"] == COMMIT
    for name, (digest, size) in FILES.items():
        assert growth["data_files"][name] == {"sha256": digest, "bytes": size}
        assert manifest["files"][name]["sha256"] == digest
        assert manifest["files"][name]["bytes"] == size
    assert growth["recovery"]["fit_days"] == [20., 40., 60.]
    assert growth["recovery"]["withheld_day"] == 90.

    from brainsmodels.growth_verification import REFERENCE_COMMIT, REFERENCE_MODEL_SHA256
    resources = json.loads((ROOT / "talk/resources.json").read_text())
    deck = resources["deck"]
    deck_path = ROOT / deck["path"]
    assert deck_path == ROOT / "talk" / deck["filename"], "Unexpected deck path"
    assert deck_path in files, "Final deck is missing from the publication contents"
    assert deck_path.stat().st_size == deck["size_bytes"], "Deck byte size differs from manifest"
    with deck_path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    assert digest == deck["sha256"], "Deck checksum differs from manifest"
    with zipfile.ZipFile(deck_path) as package:
        assert package.testzip() is None, "Damaged PPTX package"
        presentation = ET.fromstring(package.read("ppt/presentation.xml"))
        slides = presentation.find(
            "{http://schemas.openxmlformats.org/presentationml/2006/main}sldIdLst")
        assert slides is not None and len(slides) == deck["slide_count"], \
            "Deck slide count differs from manifest"
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
          f"local links; {len(notebooks)} notebooks with course navigation; "
          f"numerical references; Python syntax; "
          f"final PPTX checksum, size, package and {deck['slide_count']} slides.")


if __name__ == "__main__":
    main()
