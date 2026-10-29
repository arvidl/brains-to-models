PYTHON ?= $(firstword $(wildcard $(CURDIR)/.venv/bin/python) python3)
export PYTHONPATH := $(CURDIR)/code/src$(if $(PYTHONPATH),:$(PYTHONPATH),)
export MPLBACKEND := Agg
export PYTHONHASHSEED := 7
export MPLCONFIGDIR := $(CURDIR)/.build/matplotlib

.PHONY: figures growth-figures paper all clean-paper check check-segmentation check-examples check-growth check-growth-brainweb check-growth-widgets check-notebooks

figures:
	mkdir -p paper/figures
	$(PYTHON) -m brainsmodels.marr_levels
	$(PYTHON) -m brainsmodels.timeline
	$(PYTHON) -m brainsmodels.predictive_coding
	$(PYTHON) -m brainsmodels.tiny_transformer
	$(PYTHON) -m brainsmodels.tmaze
	$(PYTHON) -m brainsmodels.bio_vs_artificial
	$(PYTHON) -m brainsmodels.biases
	$(PYTHON) -m brainsmodels.four_views
	$(PYTHON) -m brainsmodels.dynamics
	$(PYTHON) -m brainsmodels.landscape
	$(PYTHON) -m brainsmodels.analog_waves
	$(PYTHON) -m brainsmodels.workspace
	$(PYTHON) -m brainsmodels.values
	$(PYTHON) -m brainsmodels.clinical_examples
	$(PYTHON) -m brainsmodels.clinical_appendix
	$(PYTHON) -m brainsmodels.attention_maps
	$(MAKE) growth-figures

growth-figures:
	$(PYTHON) scripts/build_growth_lessons.py

paper:
	cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

# Keep the stages ordered even when make is invoked with -j.
all: figures
	$(MAKE) paper
	$(MAKE) check

check:
	$(PYTHON) scripts/check_project.py
	$(PYTHON) scripts/check_segmentation.py
	$(PYTHON) scripts/check_paper_examples.py
	$(PYTHON) scripts/check_teaching_models.py
	$(PYTHON) scripts/check_growth.py
	$(PYTHON) scripts/check_growth_extension.py
	$(PYTHON) scripts/check_growth_widgets.py

check-growth:
	$(PYTHON) scripts/check_growth.py
	$(PYTHON) scripts/check_growth_extension.py

check-growth-brainweb:
	$(PYTHON) scripts/check_growth_extension.py --brainweb

check-growth-widgets:
	$(PYTHON) scripts/check_growth_widgets.py

check-segmentation:
	$(PYTHON) scripts/check_segmentation.py

check-examples:
	$(PYTHON) scripts/check_paper_examples.py

check-notebooks:
	$(PYTHON) scripts/check_notebooks.py

clean-paper:
	cd paper && latexmk -c
