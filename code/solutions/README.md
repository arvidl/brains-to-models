# Notebook exercises

Each of notebooks 01–07 (including `b` companions) ends with three
exercises. Solutions live here so they can be `%load`-ed from a clone:

```
%load ../solutions/01_1_drop_the_causal_mask.py
```

The same code is in the notebook’s `<details>` block. Files are
standalone (seed 7) and run from `code/solutions/` or from a notebook
cell after the setup cell.

Regenerate the `.py` files and the notebook cells with:

```bash
.venv/bin/python code/solutions/_embed.py
```

These are teaching checks, not new measurements. They do not write
`paper/figures/`.

Notebooks 00–07 also end with **Questions for reflection** (ten questions
against the two themes, for both audiences) and **Further reading**
(annotated core references). The two-theme opener is only in Further
reading. Short instructor notes (not a quiz key) reprint each question
in `code/notebooks/questions_responses.md`. Regenerate with:

```bash
.venv/bin/python code/notebooks/_questions.py
.venv/bin/python code/notebooks/_reading.py
.venv/bin/python code/notebooks/_responses.py
```
