.PHONY: install bootstrap configure check test lint run push dag evidence evidence-tools submission document publish
.DEFAULT_GOAL := install

export PATH := $(CURDIR)/.venv/bin:$(PATH)
export UV_PYTHON_INSTALL_DIR := $(CURDIR)/.python
export UV_CACHE_DIR := $(CURDIR)/.uv-cache
export UV_HTTP_TIMEOUT := 300
PYTHON := $(CURDIR)/.venv/bin/python
DVC := $(CURDIR)/.venv/bin/dvc
UV := $(CURDIR)/.bootstrap/bin/uv

.bootstrap/bin/uv:
	python3 -m venv .bootstrap
	.bootstrap/bin/python -m pip install uv==0.9.9

.venv: .bootstrap/bin/uv
	$(UV) python install 3.11.14
	$(UV) venv --python 3.11.14 .venv

install: .venv
	$(UV) pip install --python $(PYTHON) -r requirements.lock
	$(UV) pip install --python $(PYTHON) --no-deps -e .

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

document: evidence-tools
	.browser-tools/bin/python scripts/document.py

publish:
	$(PYTHON) scripts/publish.py

.browser-tools/bin/python: | .venv
	$(UV) venv --python $(PYTHON) .browser-tools

evidence-tools: .browser-tools/bin/python
	$(UV) pip install --python .browser-tools/bin/python -r requirements-evidence.txt
	PLAYWRIGHT_BROWSERS_PATH=$(CURDIR)/.browser-cache .browser-tools/bin/python -m playwright install chromium --only-shell

evidence: evidence-tools
	PLAYWRIGHT_BROWSERS_PATH=$(CURDIR)/.browser-cache .browser-tools/bin/python scripts/evidence.py
