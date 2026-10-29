"""Generate bibliographic reading lists for the background notebooks."""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
NB = ROOT / "code/notebooks"

T = {'transformer': 'https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)',
 'attention': 'https://en.wikipedia.org/wiki/Attention_(machine_learning)',
 'foundation': 'https://en.wikipedia.org/wiki/Foundation_model',
 'llm': 'https://en.wikipedia.org/wiki/In-context_learning',
 'icl': 'https://en.wikipedia.org/wiki/In-context_learning',
 'pc': 'https://en.wikipedia.org/wiki/Predictive_coding',
 'fep': 'https://en.wikipedia.org/wiki/Free_energy_principle',
 'aif': 'https://en.wikipedia.org/wiki/Active_inference',
 'rl': 'https://en.wikipedia.org/wiki/Reinforcement_learning',
 'ql': 'https://en.wikipedia.org/wiki/Q-learning',
 'backprop': 'https://en.wikipedia.org/wiki/Backpropagation',
 'hebb': 'https://en.wikipedia.org/wiki/Hebbian_theory',
 'hopfield': 'https://en.wikipedia.org/wiki/Hopfield_network',
 'attractor': 'https://en.wikipedia.org/wiki/Attractor',
 'necker': 'https://en.wikipedia.org/wiki/Necker_cube',
 'gwt': 'https://en.wikipedia.org/wiki/Global_workspace_theory',
 'iit': 'https://en.wikipedia.org/wiki/Integrated_information_theory',
 'framing': 'https://en.wikipedia.org/wiki/Framing_effect_(psychology)',
 'trolley': 'https://en.wikipedia.org/wiki/Trolley_problem',
 'turing': 'https://en.wikipedia.org/wiki/Turing_test',
 'mechint': 'https://en.wikipedia.org/wiki/Mechanistic_interpretability',
 'allostasis': 'https://en.wikipedia.org/wiki/Allostasis',
 'homeostasis': 'https://en.wikipedia.org/wiki/Homeostasis',
 'consciousness': 'https://plato.stanford.edu/entries/consciousness/',
 'embodiment': 'https://plato.stanford.edu/entries/embodied-cognition/',
 'ethics_ai': 'https://plato.stanford.edu/entries/ethics-ai/'}

READING = {'00_intro_timeline': '### Further reading\n'
                      '\n'
                      'These references provide background for the models and discussion. Distinguish '
                      'primary experiments, reviews and commentary when using them as evidence. Section '
                      'references follow the August background paper.\n'
                      '\n'
                      '1. **Marr, D. (1982).** [*Vision: A Computational Investigation into the Human '
                      'Representation and Processing of Visual '
                      'Information*](https://hesperusisbosphorus.wordpress.com/wp-content/uploads/2015/10/david-marr-1982-vision-introduction.pdf). '
                      'W. H. Freeman. Introduction, pp. 24–25 and Figure 1-4.\n'
                      '2. **Turing, A. M. (1950).** [*Computing Machinery and '
                      'Intelligence*](https://redirect.cs.umbc.edu/courses/471/papers/turing.pdf). *Mind* '
                      '59(236), 433–460.\n'
                      '3. **Russell, S., and Norvig, P. (2021).** [*Artificial Intelligence: A Modern '
                      'Approach*](https://aima.cs.berkeley.edu/). 4th ed., Pearson. Book site.\n'
                      '4. **Bommasani, R., et al. (2021).** [*On the Opportunities and Risks of Foundation '
                      'Models*](https://arxiv.org/abs/2108.07258). arXiv:2108.07258.\n'
                      '5. **Doerig, A., et al. (2023).** [*The neuroconnectionist research '
                      'programme*](https://konklab.fas.harvard.edu/Papers/Doerig_2023_NRN.pdf). *Nature '
                      'Reviews Neuroscience* 24, 431–450.\n'
                      '6. **Butlin, P., Long, R., et al. (2023).** [*Consciousness in Artificial '
                      'Intelligence: Insights from the Science of '
                      'Consciousness*](https://arxiv.org/abs/2308.08708). arXiv:2308.08708.\n',
 '01_tiny_transformer': '### Further reading\n'
                        '\n'
                        'These references provide background for the models and discussion. Distinguish '
                        'primary experiments, reviews and commentary when using them as evidence. Section '
                        'references follow the August background paper.\n'
                        '\n'
                        '1. **Vaswani, A., et al. (2017).** [*Attention Is All You '
                        'Need*](https://arxiv.org/abs/1706.03762). *NeurIPS* 30. arXiv:1706.03762.\n'
                        '2. **Brown, T. B., et al. (2020).** [*Language Models are Few-Shot '
                        'Learners*](https://arxiv.org/abs/2005.14165). *NeurIPS* 33. arXiv:2005.14165.\n'
                        '3. **Bommasani, R., et al. (2021).** [*On the Opportunities and Risks of Foundation '
                        'Models*](https://arxiv.org/abs/2108.07258). arXiv:2108.07258.\n'
                        '4. **Elhage, N., et al. (2021).** [*A Mathematical Framework for Transformer '
                        'Circuits*](https://transformer-circuits.pub/2021/framework/index.html). '
                        '*Transformer Circuits Thread*.\n'
                        '5. **Jain, S., and Wallace, B. C. (2019).** [*Attention is not '
                        'Explanation*](https://aclanthology.org/N19-1357/). *NAACL* 2019, 3543–3556.\n'
                        '6. **Schrimpf, M., et al. (2021).** [*The neural architecture of language: '
                        'Integrative modeling converges on predictive '
                        'processing*](https://www.pnas.org/doi/10.1073/pnas.2105646118). *PNAS* 118(45), '
                        'e2105646118.\n'
                        '7. **Goldstein, A., et al. (2022).** [*Shared computational principles for language '
                        'processing in humans and deep language '
                        'models*](https://doi.org/10.1038/s41593-022-01026-4). *Nature Neuroscience* 25, '
                        '369–380.\n'
                        '8. **Lindsey, J., Gurnee, W., Ameisen, E., et al. (2025).** [*On the Biology of a '
                        'Large Language '
                        'Model*](https://transformer-circuits.pub/2025/attribution-graphs/biology.html). '
                        '*Transformer Circuits Thread*.\n'
                        '9. **Ouyang, L., et al. (2022).** [*Training language models to follow instructions '
                        'with human feedback*](https://arxiv.org/abs/2203.02155). *NeurIPS* 35. '
                        'arXiv:2203.02155.\n',
 '02_predictive_coding': '### Further reading\n'
                         '\n'
                         'These references provide background for the models and discussion. Distinguish '
                         'primary experiments, reviews and commentary when using them as evidence. Section '
                         'references follow the August background paper.\n'
                         '\n'
                         '1. **Rao, R. P. N., and Ballard, D. H. (1999).** [*Predictive coding in the visual '
                         'cortex: a functional interpretation of some extra-classical receptive-field '
                         'effects*](https://homes.cs.washington.edu/~rao/Rao-Ballard-NN-1999.pdf). *Nature '
                         'Neuroscience* 2, 79–87. Author PDF.\n'
                         '2. **Friston, K. (2010).** [*The free-energy principle: a unified brain '
                         'theory?*](https://www.fil.ion.ucl.ac.uk/~karl/The%20free-energy%20principle%20A%20unified%20brain%20theory.pdf). '
                         '*Nature Reviews Neuroscience* 11, 127–138. Author PDF.\n'
                         '3. **Clark, A. (2013).** [*Whatever next? Predictive brains, situated agents, and '
                         'the future of cognitive science*](https://doi.org/10.1017/S0140525X12000477). '
                         '*Behavioral and Brain Sciences* 36(3), 181–204.\n'
                         '4. **Salvatori, T., et al. (2023).** [*A Survey on Brain-inspired Deep Learning '
                         'via Predictive Coding*](https://arxiv.org/abs/2308.07870). arXiv:2308.07870 '
                         '(journal version: *Neural Networks* 195, 108161, 2026).\n'
                         '5. **Schrimpf, M., et al. (2021).** [*The neural architecture of language: '
                         'Integrative modeling converges on predictive '
                         'processing*](https://www.pnas.org/doi/10.1073/pnas.2105646118). *PNAS* 118(45), '
                         'e2105646118.\n'
                         '6. **Seth, A. (2021).** [*Being You: A New Science of '
                         'Consciousness*](https://www.anilseth.com/being-you). Faber & Faber. Author page.\n',
 '02b_active_inference_vs_rl': '### Further reading\n'
                               '\n'
                               'These references provide background for the models and discussion. '
                               'Distinguish primary experiments, reviews and commentary when using them as '
                               'evidence. Section references follow the August background paper.\n'
                               '\n'
                               '1. **Friston, K. J., Daunizeau, J., and Kiebel, S. J. (2009).** '
                               '[*Reinforcement learning or active '
                               'inference?*](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0006421). '
                               '*PLoS ONE* 4(7), e6421.\n'
                               '2. **Sajid, N., Ball, P. J., Parr, T., and Friston, K. J. (2021).** [*Active '
                               'inference: demystified and compared*](https://arxiv.org/abs/1909.10863). '
                               '*Neural Computation* 33(3), 674–712. arXiv:1909.10863.\n'
                               '3. **Friston, K., et al. (2017).** [*Active inference: A process '
                               'theory*](https://doi.org/10.1162/NECO_a_00912). *Neural Computation* 29(1), '
                               '1–49.\n'
                               '4. **Levine, S. (2018).** [*Reinforcement learning and control as '
                               'probabilistic inference: Tutorial and '
                               'review*](https://arxiv.org/abs/1805.00909). arXiv:1805.00909.\n'
                               '5. **Heins, C., et al. (2022).** [*pymdp: A Python library for active '
                               'inference in discrete state spaces*](https://doi.org/10.21105/joss.04098). '
                               '*Journal of Open Source Software* 7(73), 4098.\n'
                               '6. **Friston, K. (2010).** [*The free-energy principle: a unified brain '
                               'theory?*](https://www.fil.ion.ucl.ac.uk/~karl/The%20free-energy%20principle%20A%20unified%20brain%20theory.pdf). '
                               '*Nature Reviews Neuroscience* 11, 127–138. Author PDF.\n',
 '03_bio_vs_artificial': '### Further reading\n'
                         '\n'
                         'These references provide background for the models and discussion. Distinguish '
                         'primary experiments, reviews and commentary when using them as evidence. Section '
                         'references follow the August background paper.\n'
                         '\n'
                         '1. **Lillicrap, T. P., Santoro, A., Marris, L., Akerman, C. J., and Hinton, G. '
                         '(2020).** [*Backpropagation and the '
                         'brain*](https://www.cs.toronto.edu/~hinton/absps/backpropandbrain.pdf). *Nature '
                         'Reviews Neuroscience* 21, 335–346. Author PDF.\n'
                         '2. **Yamins, D. L. K., and DiCarlo, J. J. (2016).** [*Using goal-driven deep '
                         'learning models to understand sensory cortex*](https://doi.org/10.1038/nn.4244). '
                         '*Nature Neuroscience* 19, 356–365.\n'
                         '3. **Doerig, A., et al. (2023).** [*The neuroconnectionist research '
                         'programme*](https://konklab.fas.harvard.edu/Papers/Doerig_2023_NRN.pdf). *Nature '
                         'Reviews Neuroscience* 24, 431–450.\n'
                         '4. **Kriegeskorte, N., Mur, M., and Bandettini, P. (2008).** [*Representational '
                         'similarity analysis — connecting the branches of systems '
                         'neuroscience*](https://www.frontiersin.org/articles/10.3389/neuro.06.004.2008/full). '
                         '*Frontiers in Systems Neuroscience* 2, 4.\n'
                         '5. **Hebb, D. O. (1949).** [*The Organization of '
                         'Behavior*](https://en.wikipedia.org/wiki/Hebbian_theory). Wiley. Entry via the '
                         'opened Hebbian-theory page.\n'
                         '6. **Vaswani, A., et al. (2017).** [*Attention Is All You '
                         'Need*](https://arxiv.org/abs/1706.03762). *NeurIPS* 30. arXiv:1706.03762.\n',
 '04_quadrants_biases': '### Further reading\n'
                        '\n'
                        'These references provide background for the models and discussion. Distinguish '
                        'primary experiments, reviews and commentary when using them as evidence. Section '
                        'references follow the August background paper.\n'
                        '\n'
                        '1. **Tversky, A., and Kahneman, D. (1981).** [*The Framing of Decisions and the '
                        'Psychology of '
                        'Choice*](https://stat.columbia.edu/~gelman/surveys.course/TverskyKahneman1981.pdf). '
                        '*Science* 211(4481), 453–458. Opened PDF.\n'
                        '2. **Binz, M., and Schulz, E. (2023).** [*Using cognitive psychology to understand '
                        'GPT-3*](https://www.pnas.org/doi/10.1073/pnas.2218523120). *PNAS* 120(6), '
                        'e2218523120.\n'
                        '3. **Russell, S., and Norvig, P. (2021).** [*Artificial Intelligence: A Modern '
                        'Approach*](https://aima.cs.berkeley.edu/). 4th ed., Pearson. Book site.\n'
                        '4. **Brown, T. B., et al. (2020).** [*Language Models are Few-Shot '
                        'Learners*](https://arxiv.org/abs/2005.14165). *NeurIPS* 33. arXiv:2005.14165.\n'
                        '5. **Jain, S., and Wallace, B. C. (2019).** [*Attention is not '
                        'Explanation*](https://aclanthology.org/N19-1357/). *NAACL* 2019, 3543–3556.\n'
                        '6. **Zhao, H., et al. (2024).** [*Explainability for Large Language Models: A '
                        'Survey*](https://arxiv.org/abs/2309.01029). *ACM TIST* 15(2). arXiv:2309.01029.\n',
 '04b_four_views_of_AI': '### Further reading\n'
                         '\n'
                         'These references provide background for the models and discussion. Distinguish '
                         'primary experiments, reviews and commentary when using them as evidence. Section '
                         'references follow the August background paper.\n'
                         '\n'
                         '1. **Russell, S., and Norvig, P. (2021).** [*Artificial Intelligence: A Modern '
                         'Approach*](https://aima.cs.berkeley.edu/). 4th ed., Pearson. Book site.\n'
                         '2. **Turing, A. M. (1950).** [*Computing Machinery and '
                         'Intelligence*](https://redirect.cs.umbc.edu/courses/471/papers/turing.pdf). *Mind* '
                         '59(236), 433–460.\n'
                         '3. **Binz, M., and Schulz, E. (2023).** [*Using cognitive psychology to understand '
                         'GPT-3*](https://www.pnas.org/doi/10.1073/pnas.2218523120). *PNAS* 120(6), '
                         'e2218523120.\n'
                         '4. **Ouyang, L., et al. (2022).** [*Training language models to follow '
                         'instructions with human feedback*](https://arxiv.org/abs/2203.02155). *NeurIPS* '
                         '35. arXiv:2203.02155.\n'
                         '5. **Bommasani, R., et al. (2021).** [*On the Opportunities and Risks of '
                         'Foundation Models*](https://arxiv.org/abs/2108.07258). arXiv:2108.07258.\n'
                         '6. **Gabriel, I. (2020).** [*Artificial intelligence, values, and '
                         'alignment*](https://arxiv.org/pdf/2001.09768). *Minds and Machines* 30, 411–437. '
                         'arXiv:2001.09768.\n',
 '05_multistable_dynamics': '### Further reading\n'
                            '\n'
                            'These references provide background for the models and discussion. Distinguish '
                            'primary experiments, reviews and commentary when using them as evidence. '
                            'Section references follow the August background paper.\n'
                            '\n'
                            '1. **Hopfield, J. J. (1982).** [*Neural networks and physical systems with '
                            'emergent collective computational '
                            'abilities*](https://redwood.berkeley.edu/wp-content/uploads/2018/08/hopfield82.pdf). '
                            '*PNAS* 79(8), 2554–2558. Opened PDF.\n'
                            '2. **Ramsauer, H., et al. (2020).** [*Hopfield Networks is All You '
                            'Need*](https://arxiv.org/abs/2008.02217). arXiv:2008.02217.\n'
                            '3. **Vaswani, A., et al. (2017).** [*Attention Is All You '
                            'Need*](https://arxiv.org/abs/1706.03762). *NeurIPS* 30. arXiv:1706.03762.\n'
                            '4. **Necker, L. A. (1832).** [*Observations on some remarkable optical '
                            'phaenomena…*](https://en.wikipedia.org/wiki/Necker_cube). Entry via the opened '
                            'Necker-cube page.\n'
                            '5. **Elhage, N., et al. (2021).** [*A Mathematical Framework for Transformer '
                            'Circuits*](https://transformer-circuits.pub/2021/framework/index.html). '
                            '*Transformer Circuits Thread*.\n'
                            '6. **Doerig, A., et al. (2023).** [*The neuroconnectionist research '
                            'programme*](https://konklab.fas.harvard.edu/Papers/Doerig_2023_NRN.pdf). '
                            '*Nature Reviews Neuroscience* 24, 431–450.\n',
 '05b_multistable_animation': '### Further reading\n'
                              '\n'
                              'These references provide background for the models and discussion. '
                              'Distinguish primary experiments, reviews and commentary when using them as '
                              'evidence. Section references follow the August background paper.\n'
                              '\n'
                              '1. **Hopfield, J. J. (1982).** [*Neural networks and physical systems with '
                              'emergent collective computational '
                              'abilities*](https://redwood.berkeley.edu/wp-content/uploads/2018/08/hopfield82.pdf). '
                              '*PNAS* 79(8), 2554–2558. Opened PDF.\n'
                              '2. **Ramsauer, H., et al. (2020).** [*Hopfield Networks is All You '
                              'Need*](https://arxiv.org/abs/2008.02217). arXiv:2008.02217.\n'
                              '3. **Necker, L. A. (1832).** [*Observations on some remarkable optical '
                              'phaenomena…*](https://en.wikipedia.org/wiki/Necker_cube). Entry via the '
                              'opened Necker-cube page.\n'
                              '4. **Friston, K. (2010).** [*The free-energy principle: a unified brain '
                              'theory?*](https://www.fil.ion.ucl.ac.uk/~karl/The%20free-energy%20principle%20A%20unified%20brain%20theory.pdf). '
                              '*Nature Reviews Neuroscience* 11, 127–138. Author PDF.\n'
                              '5. **Doerig, A., et al. (2023).** [*The neuroconnectionist research '
                              'programme*](https://konklab.fas.harvard.edu/Papers/Doerig_2023_NRN.pdf). '
                              '*Nature Reviews Neuroscience* 24, 431–450.\n',
 '05c_analog_brain_waves': '### Further reading\n'
                           '\n'
                           'These references provide background for the models and discussion. Distinguish '
                           'primary experiments, reviews and commentary when using them as evidence. Section '
                           'references follow the August background paper.\n'
                           '\n'
                           '1. **Miller, E. K., Brincat, S. L., and Roy, J. E. (2026).** [*Analog Cognition '
                           'and Consciousness*](https://doi.org/10.1523/JNEUROSCI.0711-26.2026). *J. '
                           'Neurosci.* 46(33):e0711262026. Opened PDF.\n'
                           '2. **Hopfield, J. J. (1982).** [*Neural networks and physical systems with '
                           'emergent collective computational '
                           'abilities*](https://redwood.berkeley.edu/wp-content/uploads/2018/08/hopfield82.pdf). '
                           '*PNAS* 79(8), 2554–2558. Opened PDF.\n'
                           '3. **Block, N. (1995).** [*On a confusion about a function of '
                           'consciousness*](https://divinecuration.github.io/assets/pdf/block-confusion.pdf). '
                           '*Behavioral and Brain Sciences* 18(2), 227–247. Opened PDF.\n'
                           '4. **Dehaene, S. (2014).** [*Consciousness and the '
                           'Brain*](https://www.penguinrandomhouse.com/books/310548/consciousness-and-the-brain-by-stanislas-dehaene/). '
                           'Viking. Publisher page.\n',
 '06_global_workspace': '### Further reading\n'
                        '\n'
                        'These references provide background for the models and discussion. Distinguish '
                        'primary experiments, reviews and commentary when using them as evidence. Section '
                        'references follow the August background paper.\n'
                        '\n'
                        '1. **Baars, B. J. (1988).** [*A Cognitive Theory of '
                        'Consciousness*](https://en.wikipedia.org/wiki/Global_workspace_theory). Cambridge '
                        'University Press. Entry via the opened GWT page.\n'
                        '2. **Block, N. (1995).** [*On a confusion about a function of '
                        'consciousness*](https://divinecuration.github.io/assets/pdf/block-confusion.pdf). '
                        '*Behavioral and Brain Sciences* 18(2), 227–247. Opened PDF.\n'
                        '3. **Dehaene, S. (2014).** [*Consciousness and the '
                        'Brain*](https://www.penguinrandomhouse.com/books/310548/consciousness-and-the-brain-by-stanislas-dehaene/). '
                        'Viking. Publisher page.\n'
                        '4. **Butlin, P., Long, R., et al. (2023).** [*Consciousness in Artificial '
                        'Intelligence: Insights from the Science of '
                        'Consciousness*](https://arxiv.org/abs/2308.08708). arXiv:2308.08708.\n'
                        '5. **Gurnee, W., Sofroniew, N., Pearce, A., et al. (2026).** [*Verbalizable '
                        'Representations Form a Global Workspace in Language '
                        'Models*](https://transformer-circuits.pub/2026/workspace). *Transformer Circuits '
                        'Thread*.\n'
                        '6. **Tononi, G., Boly, M., Massimini, M., and Koch, C. (2016).** [*Integrated '
                        'information theory: from consciousness to its physical '
                        'substrate*](https://en.wikipedia.org/wiki/Integrated_information_theory). *Nature '
                        'Reviews Neuroscience* 17, 450–461.\n'
                        '7. **Seth, A. (2021).** [*Being You: A New Science of '
                        'Consciousness*](https://www.anilseth.com/being-you). Faber & Faber. Author page.\n',
 '07_values_probing': '### Further reading\n'
                      '\n'
                      'These references provide background for the models and discussion. Distinguish '
                      'primary experiments, reviews and commentary when using them as evidence. Section '
                      'references follow the August background paper.\n'
                      '\n'
                      '1. **Greene, J. D., Sommerville, R. B., Nystrom, L. E., Darley, J. M., and Cohen, J. '
                      'D. (2001).** [*An fMRI Investigation of Emotional Engagement in Moral '
                      'Judgment*](https://pubmed.ncbi.nlm.nih.gov/11557895/). *Science* 293(5537), '
                      '2105–2108. PubMed abstract.\n'
                      '2. **Gabriel, I. (2020).** [*Artificial intelligence, values, and '
                      'alignment*](https://arxiv.org/pdf/2001.09768). *Minds and Machines* 30, 411–437. '
                      'arXiv:2001.09768.\n'
                      '3. **Ouyang, L., et al. (2022).** [*Training language models to follow instructions '
                      'with human feedback*](https://arxiv.org/abs/2203.02155). *NeurIPS* 35. '
                      'arXiv:2203.02155.\n'
                      '4. **Bai, Y., et al. (2022).** [*Constitutional AI: Harmlessness from AI '
                      'Feedback*](https://arxiv.org/abs/2212.08073). arXiv:2212.08073.\n'
                      '5. **Tversky, A., and Kahneman, D. (1981).** [*The Framing of Decisions and the '
                      'Psychology of '
                      'Choice*](https://stat.columbia.edu/~gelman/surveys.course/TverskyKahneman1981.pdf). '
                      '*Science* 211(4481), 453–458. Opened PDF.\n'
                      '6. **Binz, M., and Schulz, E. (2023).** [*Using cognitive psychology to understand '
                      'GPT-3*](https://www.pnas.org/doi/10.1073/pnas.2218523120). *PNAS* 120(6), '
                      'e2218523120.\n'
                      '7. **Seth, A. (2021).** [*Being You: A New Science of '
                      'Consciousness*](https://www.anilseth.com/being-you). Faber & Faber. Author page.\n'}

def upsert_cell(nb_path: Path, source: str) -> None:
    nb = json.loads(nb_path.read_text())
    marker = "### Further reading\n"
    cells = []
    for cell in nb["cells"]:
        src = "".join(cell.get("source", []))
        if cell.get("id") == "reading001":
            continue
        if cell.get("cell_type") == "markdown" and src.startswith(marker):
            continue
        cells.append(cell)
    cells.append(
        {
            "cell_type": "markdown",
            "id": "reading001",
            "metadata": {},
            "source": source.splitlines(keepends=True),
        }
    )
    nb["cells"] = cells
    nb_path.write_text(json.dumps(nb, indent=1) + "\n")


def main() -> None:
    mapping = {
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
    for key, filename in mapping.items():
        path = NB / filename
        if not path.exists():
            raise SystemExit(f"missing {path}")
        upsert_cell(path, READING[key].rstrip() + "\n")
        n = sum(line[:1].isdigit() for line in READING[key].splitlines())
        print("wrote", key, n, "entries")


if __name__ == "__main__":
    main()
