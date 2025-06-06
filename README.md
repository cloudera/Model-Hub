Jenkins jobs used in the Modelhub operations.

1. **mlx_concat_model_hub_yamls**

   **Description**: Job to concatenate individual YAML files under a release directory to generate a single YAML file which can be used in airgapped installations to generate ModelHub cards
  
   **URL** : https://master-01.jenkins.cloudera.com/job/mlx_concat_model_hub_yamls/

   **Parameters**:
    1. **PLATFORM** : Private or Public based on which paltform's individual modelhub catalog YAML files needs to be merged.
    2. **MLX_CRUD_APP_MAJOR_VERSION** : Major version of MLX_CRUD_APP (**like 1.51.0 and NOT 1.51.0-b133**) for which individual modelhub catalog YAML files needs to be merged.

   **How does the job work**:
   1. Based on the PLATFORM and MLX_CRUD_APP_MAJOR_VERSION individual files will be merged into a single YAML file and will be stroed under https://github.com/cloudera/Model-Hub/tree/main/models/airgapped/.
   2. The naming convention of the concatenated file will be **<MLX_CRUD_APP_MAJOR_VERSION>_concatenated.yaml**


2. **mlx_copy_modelhub_spec**

   **Description**: Job to push ModelHub changes to mlx-crud-app repo. A few customers might have restriction on accessing external github repositires so the mlx-crud-app will use the modelhub catalog files from local repo instead of external github repo. We use this job to keep both external github and mlx-crud-app master branch to be in sync.

   **URL**: https://master-01.jenkins.cloudera.com/job/mlx_copy_modelhub_spec/

   **Parameters**:
   1. **DSE_TICKET**: DSE ticket number corresponding to this copy operation
   2. **PLATFORM**: Private or Public for which the modelhub catalog files needs to be synced in mlx-crud-app
   3. **MLX_CRUD_APP_MAJOR_VERSION**: Major version of MLX_CRUD_APP (**like 1.51.0 and NOT 1.51.0-b133**) for which modelhub catalog YAML files needs to be copied.
  
   **How does the job work**:
   For the choosen platform the job will copy over all the content from Model-Hub github repo to Cloudera's mlx-crud-app repo. Any changes in the YAML files will be directly merged to master branch of mlx-crud-app repo.

    **Note**
    1. YAML files for a chosen platform will be copied over to master branch of mlx-crud-app. If needed we need to cherry-pick the commit to release branch.
  
  3. **mlx_copy_modelhub_manifest_to_model_registry**

   **Description**: Job to push changes in model-registry manifest to model-registry repo. We use this job to keep both external github and model-registry manifest master branch to be in sync.

   **URL**: https://master-01.jenkins.cloudera.com/job/mlx_copy_modelhub_manifest_to_model_registry/

   **Parameters**:
   1. **registyVersion**: model registry version for which manifest YAML files needs to be copied.
  
   **How does the job work**:
   For the choosen registyVersion the job will copy over all the content from Model-Hub/manifest/registyVersion github repo to Cloudera's model-registry repo. Any changes in the YAML files will be directly merged to master branch of mlx-crud-app repo.

    **Note**
    1. YAML files will be copied over to master branch of mlx-crud-app. If needed we need to cherry-pick the commit to release branch.

