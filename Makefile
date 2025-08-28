SHELL := bash
.ONESHELL:
.DELETE_ON_ERROR:

PYTHON ?= python3
VENV := .venv

all: setup

tfidf:
	python -m businessnlp.tfidf

bm25:
	python -m businessnlp.bm25

embeddings:
	python -m businessnlp.embeddings

semantics:
	python -m businessnlp.semantics

.PHONY: setup
setup: $(VENV)/.sentinel
	@echo "dependencies installed"

$(VENV)/.sentinel:
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -r requirements.txt
	touch $(VENV)/.sentinel

.PHONY: clean
clean:
	rm -rf $(VENV)
	rm -rf **/__pycache__