"""Execute selected notebooks in a disposable project copy, preserving source files."""

import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
DEFAULTS = (
    "08_attention_maps.ipynb", "09_segmentation_reflection.ipynb",
    "10_two_key_attention.ipynb", "11_shortcut_dependence.ipynb",
    "12_hopfield_12_units.ipynb",
)


def main():
    names = (sorted(path.name for path in (ROOT / "code/notebooks").glob("*.ipynb"))
             if sys.argv[1:] == ["--all"] else sys.argv[1:] or DEFAULTS)
    for name in names:
        if Path(name).name != name or not (ROOT / "code/notebooks" / name).is_file():
            raise SystemExit(f"Expected a filename from code/notebooks: {name}")
    output = ROOT / ".build/notebook-checks"
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="brainsmodels-notebooks-") as temporary:
        scratch = Path(temporary)
        project = scratch / "project"
        for directory in ("code", "paper"):
            shutil.copytree(ROOT / directory, project / directory,
                            ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"))
        shutil.copy2(ROOT / "pyproject.toml", project / "pyproject.toml")
        kernel = scratch / "jupyter/kernels/brainsmodels-check"
        kernel.mkdir(parents=True)
        (kernel / "kernel.json").write_text(json.dumps({
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "Brains to Models check", "language": "python",
        }))
        os.environ.update({
            "JUPYTER_PATH": str(scratch / "jupyter"),
            "JUPYTER_RUNTIME_DIR": str(scratch / "runtime"),
            "MPLCONFIGDIR": str(project / ".build/matplotlib"),
            "MPLBACKEND": "Agg", "PYTHONHASHSEED": "7",
            "PYTHONPATH": str(project / "code/src"),
            "IPYTHONDIR": str(scratch / "ipython"),
        })
        for name in names:
            notebook = nbformat.read(project / "code/notebooks" / name, as_version=4)
            notebook.metadata.pop("widgets", None)
            print(f"Executing {name}", flush=True)
            NotebookClient(notebook, timeout=600, kernel_name="brainsmodels-check",
                           resources={"metadata": {"path": str(project)}}).execute()
            nbformat.validate(notebook)
            nbformat.write(notebook, output / name)
            print(f"PASS: {name}", flush=True)
        generated = project / ".build"
        if generated.exists():
            shutil.copytree(generated, output / "generated", dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns("matplotlib"))
    print(f"Executed {len(names)} notebooks; results: {output}")


if __name__ == "__main__":
    main()
