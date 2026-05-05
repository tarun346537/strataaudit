PYTHON ?= python
ARTIFACT_DIR ?= outputs

.DEFAULT_GOAL := test
.PHONY: setup lint typecheck test run sample benchmark clean

setup:
	"$(PYTHON)" -m pip install -e ".[dev,calibration]" -c constraints/research.txt

lint:
	"$(PYTHON)" -m ruff check .
	"$(PYTHON)" -m ruff format --check .

typecheck:
	"$(PYTHON)" -m mypy strataaudit --cache-dir "$(ARTIFACT_DIR)/mypy-cache"

test:
	"$(PYTHON)" -m pytest -q --cov=strataaudit --cov-report=term --cov-report="xml:$(ARTIFACT_DIR)/coverage.xml" --junitxml="$(ARTIFACT_DIR)/pytest.xml"

run:
	"$(PYTHON)" -m strataaudit.demo --output "$(ARTIFACT_DIR)/demo.json"

sample:
	"$(PYTHON)" scripts/generate_sample_data.py --worlds 30 --output-dir "$(ARTIFACT_DIR)/fixtures"

benchmark:
	"$(PYTHON)" -m strataaudit.regression_gate --output "$(ARTIFACT_DIR)/regression-gate.json"

clean:
	"$(PYTHON)" scripts/clean.py
