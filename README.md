# catflow-inference

Inference service for an object recognition pipeline

# Setup

* Install [pre-commit](https://pre-commit.com/#install) in your virtualenv. Run
`pre-commit install` after cloning this repository.
* `poetry install`

# Develop

```
export YOLO_WEIGHTS=/path/to/weights.pt
export YOLO_THRESHOLD=0.6
uvicorn catflow_inference.main:app --reload
```

## Test

```
poetry run pytest
```

## Format/lint

These are handled by pre-commit hooks.

```
poetry run black .
poetry run ruff check .
```

# Build

```
poetry build
poetry export --without-hashes --format=requirements.txt > requirements.txt
docker build -t iank1/catflow_inference:latest .
```

Note that the docker build step will currently fail if there is more than one wheel version in `dist/`


See [catflow-docker](https://github.com/iank/catflow-docker) for `docker-compose.yml`
