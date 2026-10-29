"""Append Questions for reflection cells to notebooks 00–07 (including b).

Five questions per notebook for the biomedical and computational reading
paths. Inserted immediately before Further reading.

Regenerate with:  .venv/bin/python code/notebooks/_questions.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NB = ROOT / "code" / "notebooks"

INTRO = """\
### Questions for reflection

Use these questions to distinguish what the calculation shows from the assumptions and interpretations attached to it. Write an answer before consulting the instructor notes.
"""


def qs(*items: str) -> str:
    lines = [INTRO, ""]
    for i, q in enumerate(items, start=1):
        lines.append(f"{i}. {q}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


ITEMS = {'00_intro_timeline': ('What does agreement at one of Marr’s levels establish, and what remains open?',
                       'Choose one task and describe it at all three levels.',
                       'What evidence would distinguish a useful brain metaphor from a proposed '
                       'mechanism?',
                       'How do bodily regulation and experience relate to Marr’s framework?',
                       'What does filtering the timeline change?'),
 '01_tiny_transformer': ('What does next-character prediction measure in this experiment?',
                         'Why is the causal mask necessary for left-to-right prediction?',
                         'How would you compare this Transformer fairly with a unigram or bigram model?',
                         'Does a concentrated output distribution imply a calibrated belief?',
                         'What would be needed to interpret a head as a functional component?'),
 '02_predictive_coding': ('What differs between inference and learning in this model?',
                          'What does decreasing reconstruction error establish?',
                          'How does whitening affect the experiment?',
                          'What is a fair comparison of one-level and two-level models?',
                          'How could a cortical interpretation be tested?'),
 '02b_active_inference_vs_rl': ('What does each agent know before the first trial?',
                                'Can a reward-only planner value visiting the cue?',
                                'What changes when the planning horizon is shortened?',
                                'How should the Q-update and cue-first metric be checked?',
                                'What conclusions about organisms follow from this comparison?'),
 '03_bio_vs_artificial': ('Which factors differ between the two learners?',
                          'What does the XOR result establish?',
                          'How does a membrane leak parameter differ from depth?',
                          'Could either model learn continually?',
                          'What would a meaningful energy comparison require?'),
 '04_quadrants_biases': ('What does training on the designed corpus demonstrate?',
                         'Are the main probes held-out sentence templates?',
                         'Is the learned association in the data or in the parameters?',
                         'Does a new wording require new vocabulary?',
                         'What does the balanced control establish?'),
 '04b_four_views_of_AI': ('What do the two axes of the Russell–Norvig grid distinguish?',
                          'What should be specified in an imitation test?',
                          'How could a generated reasoning trace be evaluated?',
                          'What belongs in an evaluation of rational action?',
                          'Which claims in the grid are hypotheses rather than results?'),
 '05_multistable_dynamics': ('When is Hopfield energy guaranteed not to increase?',
                             'Does energy descent guarantee the intended memory?',
                             'What does the double-well example model?',
                             'In what sense is modern Hopfield retrieval related to attention?',
                             'What would turn the analogy into a biological test?'),
 '05b_multistable_animation': ('What does a valley in this drawing represent?',
                               'Why can a numerical descent step increase the energy?',
                               'What changes when the initial point changes?',
                               'What does the interactive view add?',
                               'What would be needed to infer a landscape from data?'),
 '05c_analog_brain_waves': ('What is specified rather than learned in the stencil example?',
                            'What does combining two gates demonstrate?',
                            'What does moving the open window change?',
                            'How does this relate to Transformer attention?',
                            'What biological evidence would support the analogy?'),
 '06_global_workspace': ('Which workspace operations are implemented?',
                         'What does the threshold comparison establish?',
                         'What intervention would test the role of broadcast?',
                         'What can this model say about subjective experience?',
                         'How should language-model workspace findings be compared with the toy?'),
 '07_values_probing': ('Where do the switch and footbridge completion patterns come from?',
                       'Does this figure contain measured human choice rates?',
                       'How are equivalent consequences tested?',
                       'What does balancing the corpus tell us about values?',
                       'Would adding preference training establish moral understanding?')}

QUESTIONS: dict[str, str] = {key: qs(*items) for key, items in ITEMS.items()}

NOTEBOOKS: dict[str, str] = {
    "00_intro_timeline": "00_intro_timeline.ipynb",
    "01_tiny_transformer": "01_tiny_transformer.ipynb",
    "02_predictive_coding": "02_predictive_coding.ipynb",
    "02b_active_inference_vs_rl": "02b_active_inference_vs_rl.ipynb",
    "03_bio_vs_artificial": "03_bio_vs_artificial.ipynb",
    "04_quadrants_biases": "04_quadrants_biases.ipynb",
    "04b_four_views_of_AI": "04b_four_views_of_AI.ipynb",
    "05_multistable_dynamics": "05_multistable_dynamics.ipynb",
    "05b_multistable_animation": "05b_multistable_animation.ipynb",
    "05c_analog_brain_waves": "05c_analog_brain_waves.ipynb",
    "06_global_workspace": "06_global_workspace.ipynb",
    "07_values_probing": "07_values_probing.ipynb",
}

TITLES: dict[str, str] = {
    "00_intro_timeline": "00 — Brain metaphors and AI milestones",
    "01_tiny_transformer": "01 — Tiny transformer",
    "02_predictive_coding": "02 — Predictive coding",
    "02b_active_inference_vs_rl": "02b — T-maze (active inference vs RL)",
    "03_bio_vs_artificial": "03 — Local spikes vs SGD",
    "04_quadrants_biases": "04 — Framing as next-token mass",
    "04b_four_views_of_AI": "04b — Four views of AI",
    "05_multistable_dynamics": "05 — Hopfield / Necker",
    "05b_multistable_animation": "05b — Energy landscape",
    "05c_analog_brain_waves": "05c — Analog brain waves",
    "06_global_workspace": "06 — Global workspace",
    "07_values_probing": "07 — Values / trolley",
}


def upsert_cell(nb_path: Path, source: str) -> None:
    nb = json.loads(nb_path.read_text())
    q_marker = "### Questions for reflection\n"
    r_marker = "### Further reading\n"
    body: list[dict] = []
    reading: list[dict] = []
    for cell in nb["cells"]:
        src = "".join(cell.get("source", []))
        if cell.get("id") == "questions001":
            continue
        if cell.get("cell_type") == "markdown" and src.startswith(q_marker):
            continue
        if cell.get("id") == "reading001" or (
            cell.get("cell_type") == "markdown" and src.startswith(r_marker)
        ):
            reading.append(cell)
            continue
        body.append(cell)
    body.append(
        {
            "cell_type": "markdown",
            "id": "questions001",
            "metadata": {},
            "source": source.splitlines(keepends=True),
        }
    )
    body.extend(reading)
    nb["cells"] = body
    nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")


def main() -> None:
    for key, filename in NOTEBOOKS.items():
        path = NB / filename
        if not path.exists():
            raise SystemExit(f"missing {path}")
        text = QUESTIONS[key]
        n = sum(
            1
            for line in text.splitlines()
            if line.startswith("10. ") or (len(line) > 3 and line[1] == "." and line[0].isdigit())
        )
        if n != len(ITEMS[key]):
            raise SystemExit(f"{key}: expected {len(ITEMS[key])} questions, found {n}")
        upsert_cell(path, text)
        print("wrote", key, n, "questions")


if __name__ == "__main__":
    main()
