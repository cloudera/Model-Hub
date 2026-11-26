# Cloudera Model Hub

A curated catalog of NVIDIA NIM microservices — production-ready AI models spanning language, vision, speech, and more — optimized for deployment with [Cloudera AI Inference Service](https://www.cloudera.com/products/machine-learning/ai-inference-service.html).

## Browse the catalog

**[→ Open the Model Hub catalog](https://cloudera.github.io/Model-Hub/)**

Models can be imported directly into the Cloudera AI Registry and deployed with a single click via Cloudera AI Inference Service.

## Catalog YAML vs manifests

This repo ships two related but separate artifact types:

| Artifact | Location | Purpose |
|---|---|---|
| **Catalog descriptors** | `models/` + `models-order/` | Product metadata for Model Hub: display names, categories, variants, license links, NGC and Hugging Face source hints, optimization profiles. Consumed by Model Hub UIs and registry catalog APIs. |
| **Manifests** | `manifests/<registryVersion>/...` | Download manifests bundled into model-registry for external import (NGC and Hugging Face). Version folders use the **model-registry release** (e.g. `1.19.0`). |

Editing catalog YAML does not update manifests, and vice versa. Both may need to change when adding a model, depending on the release path.

## Model Hub APIs

Catalog YAML in this repo is not served directly to production clusters (except the public GitHub Pages site). Runtime APIs merge `models/` + `models-order/` and join visibility state from a database.

### CDP (mlx-crud-app control plane)

| API | Method / path | Role |
|---|---|---|
| `GetModelHubSpec` | POST `/api/v1/ml/getModelHubSpec` | Returns merged catalog YAML (GitHub `cloudera/Model-Hub` when reachable, else bundled copies under `service/resources/models/`). Platform: `public`, `private`, or `csk`. |
| `listConfiguredModels` | POST `/api/v1/ml/listConfiguredModels` | Lists models with enable/disable and order from control-plane Postgres (`models` table). |
| `configureModels` | POST `/api/v1/ml/configureModels` | Admin: update which catalog entries are visible and their order. |

Model import into a tenant registry uses the **model-registry** API (e.g. POST `/api/v2/models` on the registry URL), not these mlx catalog APIs.

### AWC / CAIR (model-registry v3)

Base path: **`/api/v3`**. Catalog endpoints read bundled files synced from this repo into the registry image (`/modelhub`, `/modelhub-order`) and join **`modelhub_config`** in the registry database.

| Operation | Method / path | Role |
|---|---|---|
| `GetModelHubModels` | GET `/api/v3/modelhub/models` | Merged catalog for the UI. Query: `source`, `category`, `filter`, `include_disabled`. Response includes `modelhub_version` (registry release catalog version). Only entries present in `modelhub_config` are returned. |
| `GetModelHubConfig` | GET `/api/v3/modelhub/config` | Visibility/order rows. Query: `source`, `category`, `filter` (`ENABLED` \| `DISABLED` \| `ALL`). |
| `UpdateModelHubConfig` | PUT `/api/v3/modelhub/config` | Admin: upsert enabled flag and order by `(modelhub_id, source)`. |
| `CreateModel` + `modelhub_import` | POST `/api/v3/models` | Register a model by resolving `modelhub_import` (`model_hub_id`, `variant_id`, `profile_id`) against the bundled catalog, then running the normal NGC/HF import path. Mutually exclusive with `create_model_version_request_payload` on the same request. |

**Not the Model Hub catalog:** GET `/api/v3/models` lists **models registered in this registry instance** (user-created entries), not the product catalog. Use GET `/api/v3/modelhub/models` for catalog browsing.

Import jobs resolve manifests synced from `manifests/<registryVersion>/` in this repo into `model-registry/manifests/` (registry download code for NGC and Hugging Face).

### CDP vs AWC (where catalog bytes and APIs live)

| Concern | CDP (mlx-crud-app) | AWC (model-registry v3) |
|---|---|---|
| Catalog source | Merged YAML at runtime (GitHub + bundled `service/resources/models/`) | Bundled `modelhub/` + `modelhub-order/` in the registry image only |
| Catalog version | Version subdirs under `models/{platform}/<version>/` (historically MLX control-plane labels on CDP; on AWC use the **model-registry release** semver, e.g. `1.19.0`) | Same semver as the registry release (`config.version`, e.g. `1.19.0-b666` → catalog dir `1.19.0`); no fallback to another release folder |
| Visibility / ordering | Control-plane Postgres `models` table | Registry Postgres `modelhub_config`; seeded on startup from bundled `merge-order.yaml` |
| Catalog APIs | gRPC-backed POST `/api/v1/ml/getModelHubSpec`, `listConfiguredModels`, `configureModels` | REST GET/PUT `/api/v3/modelhub/*` |
| Import from catalog | Client → registry create/import APIs | POST `/api/v3/models` with `modelhub_import` |
| Manifests | Jenkins → `model-registry/manifests/` | Same |

## Repository structure

```
models/public/              # Catalog descriptors (CDP public cloud)
models/private/             # Catalog descriptors (CDP private cloud)
models/csk/                 # Catalog descriptors (CSK / AWC)
models-order/public/        # merge-order.yaml — catalog merge sequence
models-order/private/
models-order/csk/
manifests/<registryVersion>/...   # Import manifests per model-registry release
airgap-scripts/             # Scripts for airgapped deployments
```

Each platform tree can be laid out in two ways:

- **Top-level `*.yaml`** — used by the public GitHub Pages catalog and as non-versioned filenames in some CDP bundles.
- **`<version>/` subdirectories** — a **frozen snapshot** of catalog files for a product release. The directory name is the release label consumers resolve against (model-registry release on AWC; MLX control-plane major on legacy CDP copies). Pair with `models-order/<platform>/<version>/merge-order.yaml`.

Runtime APIs **read** merged catalog bytes; they do **not** author new catalog entries. Changing which models exist in the product catalog, or their variants and metadata, is done by editing YAML in this repo and shipping a new bundle (Jenkins sync + registry/mlx image). APIs only control **visibility**, **order**, and **import into a tenant registry** (see below).

The public GitHub Pages catalog reads from `models/public/` (and `models-order/public/`) at runtime.

## Adding or updating catalog content

**New or changed catalog entries (source of truth = this repo):**

1. Add or edit descriptor YAML under the right platform (`models/public/`, `models/private/`, or `models/csk/`) using `utils/base_model.yaml`.
2. Add the filename to the matching `merge-order.yaml` (top-level or under the `<version>/` directory you ship).
3. For model-registry releases, place files under `models/<platform>/<registryVersion>/` (e.g. `models/csk/1.19.0/`) so they align with `manifests/<registryVersion>/`.
4. Add or update manifest files under `manifests/<registryVersion>/...` when import needs new download metadata.
5. Run downstream sync (see [Downstream sync](#downstream-sync)) and cut a new mlx-crud-app or model-registry build as applicable.

**Public site only:** commit to `main` — GitHub Pages picks up `models/public/` with no build step.

**After a bundle is deployed (API-based, no catalog YAML edits):**

| Action | CDP | AWC |
|---|---|---|
| Show/hide models, change list order | POST `/api/v1/ml/configureModels` | PUT `/api/v3/modelhub/config` |
| Browse merged catalog | POST `/api/v1/ml/getModelHubSpec` / `listConfiguredModels` | GET `/api/v3/modelhub/models` |
| Import into a registry | POST `/api/v2/models` (registry URL) | POST `/api/v3/models` with `modelhub_import` |

## Downstream sync

See [OPERATIONS.md](OPERATIONS.md) for Jenkins job URLs and parameters.

### Jenkins jobs

| Job | Copies | Target repo |
|---|---|---|
| `mlx_copy_modelhub_spec` | Catalog YAML (`models/`, `models-order/`) for `Public` or `Private` | `mlx-crud-app` (`master`) |
| `mlx_copy_modelhub_manifest_to_model_registry` | `manifests/<registryVersion>/` | `model-registry/manifests/` |
| `mlx_concat_model_hub_yamls` | Concatenated airgap catalog file | `models/airgapped/` in this repo |

There is no Jenkins job in OPERATIONS.md yet for catalog → `model-registry/modelhub/`; that sync is manual or release-pipeline owned until a job is added (same source paths as below).

### Copy paths

| Consumer | What is copied from Model-Hub | Notes |
|---|---|---|
| **mlx-crud-app** (CDP) | `models/{public\|private}/` → `service/resources/models/{platform}/`; `models-order/...` → `service/resources/models/models-order/{platform}/` | Job: `mlx_copy_modelhub_spec`. Bundled fallback when GitHub is unreachable. |
| **model-registry** (manifests) | `manifests/` → `model-registry/manifests/` | Job: `mlx_copy_modelhub_manifest_to_model_registry` (`registryVersion`). |
| **model-registry** (catalog bundle) | `models/<platform>/` → `model-registry/modelhub/<platform>/`; `models-order/<platform>/` → `model-registry/modelhub-order/<platform>/` | Image build (`COPY modelhub`, `COPY modelhub-order`). For a registry release, the catalog **version directory** (e.g. `csk/1.19.0/`) and `merge-order.yaml` must match that release’s semver—the same line as `manifests/<registryVersion>/`. If that bundle is absent, v3 serves an **empty** catalog for that version. |

AWC / CSK catalog content lives under `models/csk/` and `models-order/csk/`.

## GitHub Pages setup

The catalog is served directly from `index.html` at the repo root. It fetches the YAML files at runtime.

## Internal operations

See [OPERATIONS.md](OPERATIONS.md) for Jenkins jobs used to sync content into `mlx-crud-app` and `model-registry`.
