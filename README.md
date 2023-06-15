# catflow-inference

Inference service for an object recognition pipeline

# Develop

```
poetry install

export YOLO_WEIGHTS=/path/to/weights.pt
export YOLO_THRESHOLD=0.6
uvicorn catflow_inference.main:app --reload
```

## Test

```
poetry run pytest
```

## Format/lint

```
poetry run black .
poetry run ruff check .
```



# Build

```
poetry build
poetry export --without-hashes --format=requirements.txt > requirements.txt
docker build -t iank1/catflow_inference:v0.1.0 .
```

See [catflow-docker](https://github.com/iank/catflow-docker) for `docker-compose.yml`

# TODO:

* CI
* API docs
* /status
