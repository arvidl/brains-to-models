# Notebook exercises

Each of notebooks 01–07 (including lettered companions) ends with three
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

Notebooks 00–07 also include five **Questions for reflection** and a concise
**Further reading** list. Instructor discussion notes reprint each question in
[questions_responses.md](../notebooks/questions_responses.md). These are sample
answers, with scope and assumptions stated. Regenerate the questions, reading
lists and notes with:

```bash
.venv/bin/python code/notebooks/_questions.py
.venv/bin/python code/notebooks/_reading.py
.venv/bin/python code/notebooks/_responses.py
```
