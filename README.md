# Cloudera Model Hub

A curated catalog of NVIDIA NIM microservices — production-ready AI models spanning language, vision, speech, and more — optimized for deployment with [Cloudera AI Inference Service](https://www.cloudera.com/products/machine-learning/ai-inference-service.html).

## Browse the catalog

**[→ Open the Model Hub catalog](https://cloudera.github.io/Model-Hub/)**

Models can be imported directly into the Cloudera AI Registry and deployed with a single click via Cloudera AI Inference Service.

## Repository structure

```
models/public/          # Model descriptor YAMLs (latest)
models/private/         # Private cloud variants
models-order/public/    # Merge order defining catalog sequence
manifests/              # Model Registry manifest versions
airgap-scripts/         # Scripts for airgapped deployments
```

## Adding or updating a model

1. Add or edit the model YAML in `models/public/` following the schema in `utils/base_model.yaml`.
2. Add the filename to `models-order/public/merge-order.yaml` in the desired position.
3. Commit to `main` — the catalog page picks up changes immediately, no build step required.

## GitHub Pages setup

The catalog is served directly from `index.html` at the repo root. It fetches the YAML files at runtime.

## Internal operations

See [OPERATIONS.md](OPERATIONS.md) for Jenkins jobs used to sync content into `mlx-crud-app` and `model-registry`.
