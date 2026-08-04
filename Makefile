PYTHON := .venv/bin/python
RESULTS := experiment/results

.PHONY: all validate analyze figures paper supplement word site deliverables test clean-latex

all: validate analyze figures paper supplement

validate:
	cd experiment && ../$(PYTHON) -B validate_research_results.py \
		--results results --require-final

analyze: validate
	cd experiment && ../$(PYTHON) -B analyze_research_results.py \
		--results results --bootstrap 10000

figures: analyze
	cd experiment && ../$(PYTHON) -B generate_research_figures.py \
		--results results \
		--output ../manuscript/figures \
		--supplement-output ../manuscript/supplement_figures

paper: analyze figures
	cd manuscript && latexmk -pdf -interaction=nonstopmode -halt-on-error manuscript.tex

supplement: analyze figures
	cd manuscript && latexmk -pdf -interaction=nonstopmode -halt-on-error supplement.tex

word: paper
	cd manuscript && ../$(PYTHON) -B build_docx.py --output manuscript.docx

site: word supplement
	$(PYTHON) -B paper_explainer/build_site.py

deliverables: all word site

test:
	cd experiment && ../$(PYTHON) -m unittest discover -s tests -v

clean-latex:
	cd manuscript && latexmk -c manuscript.tex && latexmk -c supplement.tex
