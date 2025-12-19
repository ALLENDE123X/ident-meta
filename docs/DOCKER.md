# Docker-based Development

This project uses Docker for reproducible builds and testing.

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed
- Docker Compose (included with Docker Desktop)

## Quick Start

```bash
# Build the container
docker-compose build

# Verify PySINDy integration
docker-compose run weakident

# Run all tests
docker-compose run test

# Run PySINDy-specific tests
docker-compose run test-pysindy
```

## Available Services

| Service | Description | Command |
|---------|-------------|---------|
| `weakident` | Verify method registration | `docker-compose run weakident` |
| `test` | Run all tests | `docker-compose run test` |
| `test-pysindy` | Test PySINDy integration | `docker-compose run test-pysindy` |
| `benchmark` | Run quick benchmark | `docker-compose run benchmark` |
| `make-dataset` | Generate dataset | `docker-compose run make-dataset` |
| `shell` | Interactive bash shell | `docker-compose run shell` |

## Custom Commands

```bash
# Run any Python script
docker-compose run weakident python scripts/train_selector.py --cfg config/default.yaml

# Interactive Python
docker-compose run weakident python

# Run specific tests
docker-compose run weakident pytest tests/test_features.py -v
```

## Rebuilding

If you modify dependencies:

```bash
docker-compose build --no-cache
```

## Volumes

- `.:/app` — Source code mounted for live editing
- `./artifacts:/app/artifacts` — Persisted outputs
- `./experiments:/app/experiments` — Experiment results
