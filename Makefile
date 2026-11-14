.PHONY: install bootstrap configure check test lint run push dag submission publish
.DEFAULT_GOAL := install

export PATH := $(CURDIR)/.venv/bin:$(PATH)
PYTHON := $(CURDIR)/.venv/bin/python
DVC := $(CURDIR)/.venv/bin/dvc

.venv:
	python3 -m venv .venv

install: .venv
	$(PYTHON) -m pip install pip==25.3
	$(PYTHON) -m pip install -r requirements.lock
	$(PYTHON) -m pip install --no-deps -e .

configure:
	$(PYTHON) scripts/configure.py

bootstrap:
	$(PYTHON) scripts/bootstrap.py

check:
	$(PYTHON) scripts/check.py

test:
	$(PYTHON) -m pytest -q

lint:
	$(CURDIR)/.venv/bin/ruff check flood scripts tests

run:
	$(DVC) repro

push:
	$(DVC) push

dag:
	$(DVC) dag

submission:
	$(PYTHON) scripts/submission.py

publish:
	$(PYTHON) scripts/publish.py
