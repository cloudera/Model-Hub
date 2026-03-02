# Operations

Jenkins jobs used in Model Hub operations.

## mlx_concat_model_hub_yamls

Concatenates individual YAML files under a release directory into a single file for airgapped installations.

**URL:** https://master-01.jenkins.cloudera.com/job/mlx_concat_model_hub_yamls/

| Parameter | Description |
|---|---|
| `PLATFORM` | `Private` or `Public` |
| `MLX_CRUD_APP_MAJOR_VERSION` | Major version only, e.g. `1.51.0` not `1.51.0-b133` |

Output is written to `models/airgapped/<version>_concatenated.yaml`.

## mlx_copy_modelhub_spec

Syncs Model Hub YAML changes into the `mlx-crud-app` repo for customers that cannot access external GitHub repositories.

**URL:** https://master-01.jenkins.cloudera.com/job/mlx_copy_modelhub_spec/

| Parameter | Description |
|---|---|
| `DSE_TICKET` | Corresponding DSE ticket number |
| `PLATFORM` | `Private` or `Public` |
| `MLX_CRUD_APP_MAJOR_VERSION` | Major version, e.g. `1.51.0` |

Changes land on `master` in `mlx-crud-app`. Cherry-pick to a release branch if needed.

## mlx_copy_modelhub_manifest_to_model_registry

Syncs model-registry manifest YAMLs from this repo into the `model-registry` repo.

**URL:** https://master-01.jenkins.cloudera.com/job/mlx_copy_modelhub_manifest_to_model_registry/

| Parameter | Description |
|---|---|
| `registryVersion` | Model registry version to sync |

Changes land on `master`. Cherry-pick to a release branch if needed.
