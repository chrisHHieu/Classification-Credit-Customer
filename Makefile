PY ?= python

.PHONY: install fetch validate features train evaluate drift all test lint serve clean docker

install:
	$(PY) -m pip install -e ".[dev]"

fetch:
	$(PY) -m credit_risk fetch

validate:
	$(PY) -m credit_risk validate

features:
	$(PY) -m credit_risk features

train:
	$(PY) -m credit_risk train

evaluate:
	$(PY) -m credit_risk evaluate

drift:
	$(PY) -m credit_risk drift

all: fetch validate features train evaluate drift

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check .

serve:
	$(PY) -m credit_risk serve --port 8000

docker:
	docker build -t credit-risk:latest .

clean:
	rm -rf .pytest_cache .ruff_cache build dist src/*.egg-info
	rm -rf data/raw/* data/processed/* models/* reports/*.json reports/figures/*.png
