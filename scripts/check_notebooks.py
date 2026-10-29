"""Execute notebooks locally or simulate a fresh Colab bootstrap in each kernel.

The Colab mode checks sparse cloning, dependency setup and calculations. It does
not exercise Google's hosted service or the browser's JavaScript widget manager.
All Git fixtures and generated files stay in disposable/ignored directories.
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
DEFAULTS = (
    "08_attention_maps.ipynb", "09_segmentation_reflection.ipynb",
    "10_two_key_attention.ipynb", "11_shortcut_dependence.ipynb",
    "12_hopfield_12_units.ipynb",
    "13_verified_growth_model.ipynb",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="Execute all eighteen notebooks")
    parser.add_argument("--colab", action="store_true", help="Test the Colab setup path in fresh local kernels")
    parser.add_argument("notebooks", nargs="*", help="Filenames under code/notebooks")
    args = parser.parse_args()
    if args.all and args.notebooks:
        parser.error("Use --all or explicit filenames, not both")
    names = (sorted(path.name for path in (ROOT / "code/notebooks").glob("*.ipynb"))
             if args.all else args.notebooks or DEFAULTS)
    for name in names:
        if Path(name).name != name or not (ROOT / "code/notebooks" / name).is_file():
            raise SystemExit(f"Expected a filename from code/notebooks: {name}")
    output = ROOT / (".build/notebook-checks-colab" if args.colab else ".build/notebook-checks")
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="brainsmodels-notebooks-") as temporary:
        scratch = Path(temporary)
        project = scratch / "project"
        for directory in ("code", "paper"):
            shutil.copytree(ROOT / directory, project / directory,
                            ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"))
        shutil.copy2(ROOT / "pyproject.toml", project / "pyproject.toml")
        if args.colab:
            print("Colab bootstrap simulation; not a hosted Colab session.", flush=True)
            # Clone a snapshot of the pending files, without committing the user's repo.
            (project / "talk").mkdir()
            (project / "talk/omitted.txt").write_text("Stand-in for material excluded from Colab.\n")
            subprocess.run(["git", "init", "-q", "--initial-branch=main", str(project)], check=True)
            subprocess.run(["git", "-C", str(project), "add", "code", "paper", "pyproject.toml", "talk"], check=True)
            subprocess.run(["git", "-C", str(project), "-c", "user.name=Notebook check",
                            "-c", "user.email=notebook-check@example.invalid",
                            "-c", "commit.gpgsign=false", "commit", "-qm", "Temporary notebook fixture"], check=True)
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
            "MPLBACKEND": "module://matplotlib_inline.backend_inline",
            "PYTHONHASHSEED": "7",
            "PYTHONPATH": str(project / "code/src"),
            "IPYTHONDIR": str(scratch / "ipython"),
        })
        for name in names:
            notebook = nbformat.read(project / "code/notebooks" / name, as_version=4)
            notebook.metadata.pop("widgets", None)
            runtime = project
            if args.colab:
                runtime = scratch / ("runtime-" + Path(name).stem)
                runtime.mkdir()
                checkout = runtime / "checkout"
                os.environ.update({
                    "BRAINSMODELS_REPO_URL": project.as_uri(),
                    "BRAINSMODELS_COLAB_ROOT": str(checkout),
                    "MPLCONFIGDIR": str(runtime / "matplotlib"),
                    # Do not let the fixture's location make shared imports succeed accidentally.
                    "PYTHONPATH": "",
                })
                notebook.cells.insert(0, nbformat.v4.new_code_cell(
                    "import sys, types\n"
                    "colab = types.ModuleType('google.colab')\n"
                    "colab.output = types.SimpleNamespace(enable_custom_widget_manager=lambda: None)\n"
                    "sys.modules['google.colab'] = colab\n"
                ))
            print(f"Executing {name}", flush=True)
            NotebookClient(notebook, timeout=600, kernel_name="brainsmodels-check",
                           resources={"metadata": {"path": str(runtime)}}).execute()
            if args.colab:
                notebook.cells.pop(0)
                assert (checkout / "code/src/brainsmodels").is_dir(), name
                assert (checkout / "paper/figures/computed_values.json").is_file(), name
                assert not (checkout / "talk").exists(), "Sparse checkout included presentation material"
                for directory in (".build", "code/animations"):
                    generated = checkout / directory
                    if generated.exists():
                        shutil.copytree(generated, output / "generated" / Path(name).stem / directory,
                                        dirs_exist_ok=True,
                                        ignore=shutil.ignore_patterns("matplotlib"))
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
