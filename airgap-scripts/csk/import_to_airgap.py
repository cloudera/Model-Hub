#!/usr/bin/env python3

import argparse
import base64
import binascii
import json
import logging
import os
import re
import stat
import subprocess
import sys
import tempfile

import yaml

MISTRAL_STAGING_MANIFEST = "mistral-staging.json"
MISTRAL_LICENSE_FIELDS = (
    "access_key_id",
    "secret_access_key",
    "region",
    "bucket",
)


# courtesy : https://stackoverflow.com/questions/43765849/pyyaml-load-and-dump-yaml-file-and-preserve-tags-customtag
class SafeUnknownConstructor(yaml.constructor.SafeConstructor):
    def __init__(self):
        yaml.constructor.SafeConstructor.__init__(self)

    def construct_undefined(self, node):
        data = getattr(self, "construct_" + node.id)(node)
        datatype = type(data)
        wraptype = type("TagWrap_" + datatype.__name__, (datatype,), {})
        wrapdata = wraptype(data)
        wrapdata.tag = lambda: None
        wrapdata.datatype = lambda: None
        setattr(wrapdata, "wrapTag", node.tag)
        setattr(wrapdata, "wrapType", datatype)
        return wrapdata


class SafeUnknownLoader(SafeUnknownConstructor, yaml.loader.SafeLoader):

    def __init__(self, stream):
        SafeUnknownConstructor.__init__(self)
        yaml.loader.SafeLoader.__init__(self, stream)


class SafeUnknownRepresenter(yaml.representer.SafeRepresenter):
    def represent_data(self, wrapdata):
        tag = False
        if type(wrapdata).__name__.startswith("TagWrap_"):
            datatype = getattr(wrapdata, "wrapType")
            tag = getattr(wrapdata, "wrapTag")
            data = datatype(wrapdata)
        else:
            data = wrapdata
        node = super(SafeUnknownRepresenter, self).represent_data(data)
        if tag:
            node.tag = tag
        return node


class SafeUnknownDumper(SafeUnknownRepresenter, yaml.dumper.SafeDumper):

    def __init__(
        self,
        stream,
        default_style=None,
        default_flow_style=False,
        canonical=None,
        indent=None,
        width=None,
        allow_unicode=None,
        line_break=None,
        encoding=None,
        explicit_start=None,
        explicit_end=None,
        version=None,
        tags=None,
        sort_keys=True,
    ):

        SafeUnknownRepresenter.__init__(
            self,
            default_style=default_style,
            default_flow_style=default_flow_style,
            sort_keys=sort_keys,
        )

        yaml.dumper.SafeDumper.__init__(
            self,
            stream,
            default_style=default_style,
            default_flow_style=default_flow_style,
            canonical=canonical,
            indent=indent,
            width=width,
            allow_unicode=allow_unicode,
            line_break=line_break,
            encoding=encoding,
            explicit_start=explicit_start,
            explicit_end=explicit_end,
            version=version,
            tags=tags,
            sort_keys=sort_keys,
        )


MySafeLoader = SafeUnknownLoader
yaml.constructor.SafeConstructor.add_constructor(
    None, SafeUnknownConstructor.construct_undefined
)


def extract_last_part(input_string):
    # Split the string by '/' and return the last part
    return input_string.split("/")[-1]


def run_ngc_info_command(ID):
    logging.info(f"Start Fetching model info from NGC: {ID}")
    cmd = ["ngc", "registry", "model", "info", ID, "--format_type", "json"]

    log_subprocess_cmd(cmd)

    # Set up environment with API keys
    # env = os.environ.copy()
    # env["NGC_CLI_API_KEY"] = os.environ.get("NGC_API_KEY")
    # env["NGC_CLI_ORG"] = os.environ.get("NGC_CLI_ORG")

    # Run the command and collect output
    try:
        result = subprocess.run(cmd, capture_output=True, text=False, check=True)

        logging.info(f"Finish Fetching NGC model repo {ID}")

        # Parse the JSON output
        try:
            metadata_map = json.loads(result.stdout)
            return metadata_map, None
        except json.JSONDecodeError as e:
            logging.error(
                f"Error while Unmarshalling the NGC info command to map: {str(e)}"
            )
            log_subprocess_output(
                logging.ERROR, "ngc model info stdout", result.stdout
            )
            log_subprocess_output(
                logging.ERROR, "ngc model info stderr", result.stderr
            )
            return None, e

    except subprocess.CalledProcessError as e:
        logging.error("Error fetching model repo: %s", ID)
        logging.error("ngc model info failed: %s", redact_log_text(str(e)))
        log_subprocess_output(logging.ERROR, "Command stdout", e.stdout or b"")
        log_subprocess_output(logging.ERROR, "Command stderr", e.stderr or b"")
        err_text = redact_log_text(
            e.stderr.decode() if e.stderr else str(e)
        )
        return None, Exception(err_text)


def get_ngc_model_info(model_id, tag):
    model_metadata_map, err = run_ngc_info_command(model_id)
    if err:
        logging.error("Error fetching model repo: %s", model_id)
        logging.error("Error: %s", redact_log_text(str(err)))
        return ""
    _, err = run_ngc_info_command(model_id + ":" + tag)
    if err:
        logging.error("Error fetching model repo: %s", model_id)
        logging.error("Error: %s", redact_log_text(str(err)))
        return ""
    return model_metadata_map


def load_ngc_spec(spec_file):
    with open(spec_file, "r") as file:
        yaml_data = file.read()

    yaml_data = yaml.load(yaml_data, Loader=MySafeLoader)
    return yaml_data


def execute_ngc_download_command(repo_id, folder_location, files=None):
    """
    Execute NGC download command to download model files.

    Args:
        repo_id (str): Repository ID
        folder_location (str): Destination folder
        files (list, optional): List of files to download. If None, download the entire repo.

    Returns:
        tuple: (folder_location, error)
    """
    # Set environment variables
    # env = os.environ.copy()
    # env["NGC_CLI_API_KEY"] = os.environ.get("NGC_API_KEY")
    # env["NGC_CLI_ORG"] = os.environ.get("NGC_CLI_ORG")
    ngcPrefix = "ngc://"
    if repo_id.startswith(ngcPrefix):
        repo_id = repo_id[len(ngcPrefix) :]
    if files and len(files) > 0:
        # Download specific files
        for file in files:
            cmd = [
                "ngc",
                "registry",
                "model",
                "download-version",
                repo_id,
                "--file",
                file,
                "--dest",
                folder_location,
                "--format_type",
                "json",
            ]
            log_subprocess_cmd(cmd)

            try:
                # output = subprocess.check_output(cmd,env=env, stderr=subprocess.STDOUT)
                output = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError as e:
                logging.error("Error downloading model repo: %s", repo_id)
                logging.error("Error: %s", redact_log_text(str(e)))
                log_subprocess_output(logging.ERROR, "Command output", e.output or b"")
                return folder_location, e
    else:
        # Download entire repository
        cmd = [
            "ngc",
            "registry",
            "model",
            "download-version",
            repo_id,
            "--dest",
            folder_location,
            "--format_type",
            "json",
        ]
        log_subprocess_cmd(cmd)

        try:
            # output = subprocess.check_output(cmd, env=env, stderr=subprocess.STDOUT)
            output = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            logging.error("Error downloading model repo: %s", repo_id)
            logging.error("Error: %s", redact_log_text(str(e)))
            log_subprocess_output(logging.ERROR, "Command output", e.output or b"")
            return folder_location, Exception(redact_log_text(e.output.decode() if e.output else str(e)))

    return folder_location, None


def extract_profile_components(data, target_profile_id):
    """
    Extract repoID and src files from components for a given profileID.

    Args:
        yaml_data (str): The YAML data as a string
        target_profile_id (str): The profileID to search for

    Returns:
        dict: A dictionary containing:
            - 'found' (bool): Whether the profile was found
            - 'components' (list): List of dictionaries with 'repo_id' and 'files' for each component
    """
    result = {
        "found": False,
        "components": [],
        "ngcMetadata": None,
    }

    # Iterate through models
    for model in data.get("models", []):
        # Iterate through variants
        for variant in model.get("modelVariants", []):
            # Iterate through profiles
            for profile in variant.get("optimizationProfiles", []):
                # Check if this is the target profileID
                if profile.get("profileId") == target_profile_id:
                    result["found"] = True
                    result["ngcMetadata"] = profile.get("ngcMetadata", None)
                    # Look for ngcMetadata which contains the workspace components
                    for sha_key, metadata in profile.get("ngcMetadata", {}).items():
                        if (
                            "workspace" in metadata
                            and "components" in metadata["workspace"]
                        ):
                            for component in metadata["workspace"]["components"]:
                                component_info = {
                                    "destination": component.get("dst", ""),
                                    "repo_id": component.get("src", {}).get(
                                        "repo_id", ""
                                    ),
                                }

                                # Extract files if they exist
                                files = component.get("src", {}).get("files", [])
                                if files:
                                    # Handle both direct strings and dictionaries with name tags
                                    component_info["files"] = []
                                    for f in files:
                                        if isinstance(f, str):
                                            component_info["files"].append(f)
                                        elif isinstance(f, dict) and f.get("!name"):
                                            component_info["files"].append(
                                                f.get("!name")
                                            )

                                result["components"].append(component_info)

                    return result  # Return once the profile is found

    return result  # Return not found if we get here


def show_help():
    """Display help information and exit."""
    help_text = """
    Description: Fetches information about a model from the Hugging Face Hub and optionally downloads it,
    or uploads model artifacts to cloud storage.

    Examples:
      python script.py gpt2
      python script.py --token YOUR_HF_TOKEN --repo-id facebook/bart-large
      python script.py --download --path ~/my_models --repo-id bert-base-uncased
      python script.py --repo-type dataset --download --repo-id mnist
      python script.py --cloud aws --src /path/to/models/ --dst s3://bucket/path --recursive
      python script.py --cloud pvc --src /path/to/model/hf/meta/llama3.1 --dst s3://bucket/secured-models/hf/meta/llama3.1
      python script.py --cloud azure --src /path/to/model/hf/meta/llama3.1 --account cloudera-customer1 --container data --dst modelregistry/secured-models/hf/meta/llama3.1
      python script.py --download --repo-type mistral --model-name mistral-small-2506 --license-key-env MISTRAL_LICENSE_KEY
      # Then upload and register with source REMOTE, model_repo_type MISTRAL, repo_id mistral-small-2506.
      # Large models (e.g. mistral-large-4-0-0) need substantial local disk (on the order of 2 TB for BF16).
    """
    print(help_text)
    sys.exit(1)


# (version command, install hint) of the CLI each operation shells out to
DOWNLOAD_REQUIREMENTS = {
    "hf": (["huggingface-cli", "version"], "pip install huggingface_hub"),
    "ngc": (
        ["ngc", "version", "info"],
        "https://org.ngc.nvidia.com/setup/installers/cli",
    ),
    "mistral": (["rclone", "version"], "https://rclone.org/install/"),
}
UPLOAD_REQUIREMENTS = {
    "aws": (["aws", "--version"], "pip install awscli"),
    "pvc": (["aws", "--version"], "pip install awscli"),
    "azure": (["az", "--version"], "pip install azure-cli"),
}


def require_tool(purpose, requirement):
    """Return True if the CLI tool is installed, else print how to get it."""
    version_cmd, install_hint = requirement
    try:
        subprocess.run(
            version_cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        print(f"Error: {version_cmd[0]} is required for {purpose} but not installed.")
        print(f"Install it: {install_hint}")
        return False
    return True


def check_requirements(download_model, cloud, repo_type=None):
    """Check that the tools needed for the requested operation are installed."""
    # An unknown repo_type is reported by validate_repo_type
    if download_model and repo_type in DOWNLOAD_REQUIREMENTS:
        if not require_tool(
            f"{repo_type} downloads", DOWNLOAD_REQUIREMENTS[repo_type]
        ):
            return False
    if cloud:
        if cloud not in UPLOAD_REQUIREMENTS:
            print(f"Unsupported cloud provider: {cloud}")
            return False
        if not require_tool(f"uploading to {cloud}", UPLOAD_REQUIREMENTS[cloud]):
            return False
    return True


def validate_repo_type(repo_type):
    """Validate the repository type."""
    valid_types = ["hf", "ngc", "mistral"]
    if repo_type not in valid_types:
        print(f"Error: Invalid repository type '{repo_type}'")
        print(f"Valid types are: {', '.join(valid_types)}")
        return False
    return True


def mistral_bucket_weights_path(path_or_name):
    """Return bucket-relative path under weights/ (e.g. weights/ministral-3b-2410)."""
    path_or_name = path_or_name.strip().strip("/")
    if path_or_name.startswith("weights/"):
        return path_or_name
    return f"weights/{path_or_name}"


def load_mistral_recipe(recipe_path):
    """Load MAIS Mistral recipe fields used for licensed-bucket download."""
    if not os.path.isfile(recipe_path):
        raise FileNotFoundError(f"recipe file not found: {recipe_path}")
    with open(recipe_path, encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    if not isinstance(doc, dict):
        raise ValueError("recipe YAML must be a mapping at the top level")
    model_name = doc.get("model_name")
    if not model_name:
        raise ValueError("recipe missing required field: model_name")
    raw_weights = doc.get("weights_path") or model_name
    weights_path = mistral_bucket_weights_path(str(raw_weights))
    speculative = doc.get("speculative_model_name")
    draft_weights_path = None
    if speculative:
        draft_weights_path = mistral_bucket_weights_path(str(speculative))
    logging.info(
        "Loaded Mistral recipe %s: model_name=%r weights_path=%r draft_weights_path=%r",
        recipe_path,
        str(model_name),
        weights_path,
        draft_weights_path,
    )
    return {
        "model_name": str(model_name),
        "weights_path": weights_path,
        "draft_weights_path": draft_weights_path,
    }


def resolve_license_key(args):
    """Return the Mistral license key from the named environment variable.

    --license-key is refused so the credential is not taken from process argv.
    """
    if getattr(args, "license_key", None):
        raise ValueError(
            "refusing --license-key; set the Mistral license in an environment "
            "variable and pass --license-key-env"
        )
    if args.license_key_env:
        value = os.environ.get(args.license_key_env)
        if not value:
            raise ValueError(
                f"environment variable '{args.license_key_env}' is unset or empty"
            )
        return value
    raise ValueError("for --repo-type mistral, provide --license-key-env")


def decode_mistral_license(key):
    """Decode a base64 Mistral license key into S3 credential fields."""
    try:
        raw = base64.b64decode(key)
    except (ValueError, binascii.Error):
        raise ValueError("Mistral license key is not valid base64") from None
    try:
        creds = json.loads(raw)
    except json.JSONDecodeError:
        raise ValueError("Mistral license key does not decode to valid JSON") from None
    if not isinstance(creds, dict):
        raise ValueError("Mistral license key JSON must be an object")
    missing = [f for f in MISTRAL_LICENSE_FIELDS if not creds.get(f)]
    if missing:
        raise ValueError(
            "Mistral license key is missing required fields: " + ", ".join(missing)
        )
    return creds


def write_rclone_config(creds, directory):
    """Write a temporary rclone config with mode 0600."""
    path = os.path.join(directory, "rclone.conf")
    body = (
        "[src]\n"
        "type = s3\n"
        "provider = AWS\n"
        f"access_key_id = {creds['access_key_id']}\n"
        f"secret_access_key = {creds['secret_access_key']}\n"
        f"region = {creds['region']}\n"
    )
    mode = stat.S_IRUSR | stat.S_IWUSR
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(body)
    except Exception:
        try:
            os.unlink(path)
        except OSError:
            pass
        raise
    return path


def build_rclone_copy_cmd(
    bucket, main_weights_path, draft_weights_path, dest, conf_path
):
    """Build the rclone copy command for Mistral weights and tokenizers."""
    main_weights_path = mistral_bucket_weights_path(main_weights_path)
    cmd = [
        "rclone",
        "copy",
        f"src:{bucket}",
        dest,
        "--config",
        conf_path,
        "--progress",
        "--include",
        f"{main_weights_path}/**",
    ]
    if draft_weights_path:
        draft_weights_path = mistral_bucket_weights_path(draft_weights_path)
        cmd.extend(["--include", f"{draft_weights_path}/**"])
    cmd.extend(["--include", "tokenizers/**"])
    return cmd


def list_mistral_bucket_weight_dirs(creds):
    """List top-level directory names under weights/ in the licensed bucket."""
    with tempfile.TemporaryDirectory() as config_dir:
        conf_path = write_rclone_config(creds, config_dir)
        remote = f"src:{creds['bucket']}/weights"
        try:
            result = subprocess.run(
                ["rclone", "lsd", remote, "--config", conf_path],
                capture_output=True,
                text=True,
                check=True,
            )
        except (subprocess.SubprocessError, FileNotFoundError) as e:
            stderr = getattr(e, "stderr", None) or ""
            if stderr:
                log_subprocess_output(logging.ERROR, "rclone lsd stderr", stderr)
            raise RuntimeError(
                f"failed to list {remote} (is rclone installed and credentials valid?): {e}"
            ) from e
    names = []
    for line in result.stdout.splitlines():
        parts = line.split()
        if parts:
            names.append(parts[-1])
    return sorted(names)


def require_mistral_draft_weights(draft_weights_path, creds):
    """Refuse to produce a Mistral artifact without resolved draft weights.

    The MAIS image enables speculative decoding by default, so every staged
    artifact must include draft/eagle weights.
    """
    if draft_weights_path:
        return
    try:
        candidates = [
            d for d in list_mistral_bucket_weight_dirs(creds)
            if "draft" in d or "eagle" in d
        ]
    except RuntimeError:
        candidates = []
    raise ValueError(
        "draft weights are required: MAIS enables speculative decoding by default. "
        "Pass --recipe with speculative_model_name (from the MAIS image, "
        "/opt/ml/model/recipes/default/<model>.yaml), or --draft-model / "
        "--draft-weights-path. "
        f"Draft/eagle dirs visible in the bucket: {candidates or 'none listed'}"
    )


def write_mistral_staging_manifest(artifacts_dir, main_weights_path, draft_weights_path):
    """Record the weight prefixes so the registry import job can verify them."""
    manifest = {"main": main_weights_path, "draft": draft_weights_path}
    with open(
        os.path.join(artifacts_dir, MISTRAL_STAGING_MANIFEST), "w", encoding="utf-8"
    ) as f:
        json.dump(manifest, f)


def print_mistral_weights_missing_hint(main_weights_path):
    """Explain common licensed-bucket / recipe mismatches."""
    print(
        "\nHint: rclone copied 0 bytes for the main weights prefix. The MAIS recipe "
        "weights_path must match a directory that exists in your licensed bucket "
        "(entitlements vary by customer). Older recipes often reference retired "
        "folders — e.g. weights/mistral-small-2506 vs weights/mistral-small-2603."
    )
    print(
        "List what your license can read:\n"
        "  python3 scripts/import_to_airgap.py --list-bucket-weights "
        "--license-key-env MISTRAL_LICENSE_KEY"
    )
    print(
        "Then use the matching recipe YAML (same weights_path basename), or override:\n"
        f"  --main-weights-path {main_weights_path.replace('mistral-small-2506', 'mistral-small-2603')}"
    )


def _mistral_tree_nonempty(directory):
    if not os.path.isdir(directory):
        return False
    for _, _, files in os.walk(directory):
        if files:
            return True
    return False


def _mistral_weights_nonempty(artifacts_dir, weights_path):
    """Check weights_path (bucket-relative, e.g. weights/ministral-3b-2410)."""
    rel = weights_path.strip("/").replace("/", os.sep)
    weights_dir = os.path.join(artifacts_dir, rel)
    return _mistral_tree_nonempty(weights_dir)


def _mistral_tokenizers_nonempty(artifacts_dir):
    return _mistral_tree_nonempty(os.path.join(artifacts_dir, "tokenizers"))


def download_repo_mistral(
    repo_id,
    draft_weights_path,
    creds,
    download_path,
    main_weights_path=None,
):
    """Download Mistral model weights from the licensed S3 bucket via rclone."""
    artifacts_dir = os.path.join(download_path, "mistral", repo_id, "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    bucket = creds["bucket"]
    main_weights_path = mistral_bucket_weights_path(
        main_weights_path or repo_id
    )
    if not draft_weights_path:
        logging.error("Draft weights path is required for Mistral download")
        return False
    draft_weights_path = mistral_bucket_weights_path(draft_weights_path)

    logging.info(
        "Mistral download: bucket=%s repo_id=%r main_weights=%r draft_weights=%r dest=%s",
        bucket,
        repo_id,
        main_weights_path,
        draft_weights_path,
        artifacts_dir,
    )

    with tempfile.TemporaryDirectory() as config_dir:
        conf_path = write_rclone_config(creds, config_dir)
        cmd = build_rclone_copy_cmd(
            bucket, main_weights_path, draft_weights_path, artifacts_dir, conf_path
        )
        log_subprocess_cmd(cmd)
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            logging.error(
                "rclone copy failed for Mistral model %s (exit code %s)",
                repo_id,
                e.returncode,
            )
            return False
        except FileNotFoundError:
            logging.error(
                "rclone not found; install from https://rclone.org/install/"
            )
            return False

    if not _mistral_weights_nonempty(artifacts_dir, main_weights_path):
        logging.error(
            "No main weights under %s (expected prefix %s/)",
            artifacts_dir,
            main_weights_path,
        )
        print_mistral_weights_missing_hint(main_weights_path)
        return False
    if not _mistral_weights_nonempty(artifacts_dir, draft_weights_path):
        logging.error(
            "No draft weights under %s (expected prefix %s/)",
            artifacts_dir,
            draft_weights_path,
        )
        return False
    if not _mistral_tokenizers_nonempty(artifacts_dir):
        logging.error("No tokenizers under %s/tokenizers", artifacts_dir)
        return False
    write_mistral_staging_manifest(artifacts_dir, main_weights_path, draft_weights_path)
    logging.info("Mistral download completed for %s at %s", repo_id, artifacts_dir)
    return True


def download_repo_hf(repo_id, token, download_path):
    """Download a Hugging Face repository using huggingface-cli."""

    print(f"Downloading repository: {repo_id} to {download_path}")

    # Create necessary directories
    os.makedirs(os.path.join(download_path, "hf", repo_id, "artifacts"), exist_ok=True)

    # Download the repository
    print("Starting download with huggingface-cli...")
    try:
        cmd = [
            "huggingface-cli",
            "download",
            repo_id,
            "--local-dir",
            f"{download_path}/hf/{repo_id}/artifacts",
        ]
        if token:
            cmd.extend(["--token", token])

        log_subprocess_cmd(cmd)
        subprocess.run(cmd, check=True)
        print(f"Download completed successfully for {repo_id}")
        return True
    except subprocess.SubprocessError:
        print(f"Error: Failed to download {repo_id}")
        return False


def download_repo(
    repo_id,
    token,
    download_path,
    repo_type,
    metadata,
    draft=None,
    creds=None,
    main_weights_path=None,
):
    """Download a repository using huggingface-cli."""
    logging.info("Starting download: repo_type=%s repo_id=%s", repo_type, repo_id)
    if repo_type == "hf":
        return download_repo_hf(repo_id, token, download_path)
    elif repo_type == "mistral":
        return download_repo_mistral(
            repo_id,
            draft,
            creds,
            download_path,
            main_weights_path=main_weights_path,
        )
    elif repo_type == "ngc":
        repo_path = extract_last_part(repo_id)
        download_path = os.path.join(download_path, "ngc", repo_path, "artifacts")
        os.makedirs(download_path, exist_ok=True)
        # Implement NGC repository downloading if needed
        for component in metadata["components"]:
            print(f"Repo ID: {component['repo_id']}")
            print(f"Destination: {component['destination']}")
            _, err = execute_ngc_download_command(
                component["repo_id"], download_path, component.get("files")
            )
            if err:
                print(f"Error: Failed to download {component['repo_id']}")
                return False
    print("Finish downloading artifacts")
    return True


def get_repo_info_hf(repo_id, token):
    """Get huggingface repository metadata and save it to a file."""
    import requests

    # Prepare headers for API request
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    # Make the API request
    url = f"https://huggingface.co/api/models/{repo_id}"
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise exception for HTTP errors

        # Parse JSON and save to file
        return response.json()
    except (requests.RequestException, json.JSONDecodeError) as e:
        print(f"Failed to fetch huggingface repo metadata for the model: {str(e)}")
        return None


def get_repo_info_ngc(repo_id, spec_file):
    """Get NGC repository metadata and save it to a file."""
    # Implement NGC repository metadata fetching if needed
    spec = load_ngc_spec(spec_file)
    ngcMetadata = extract_profile_components(spec, repo_id)
    repo_id = repo_id.split(":")
    modelMetadata = get_ngc_model_info(repo_id[0], repo_id[1])
    return ngcMetadata, modelMetadata


def get_repo_info(repo_id, token, repo_type, download_path, ngc_spec):
    """Get repository metadata and save it to a file."""
    print(f"Fetching information for {repo_type} repository: {repo_id}")
    print(f"Download path: {download_path}")
    if repo_type == "hf":
        metadata_path = os.path.join(download_path, "hf", repo_id, "metadata")
        metadata = get_repo_info_hf(repo_id, token)
        os.makedirs(metadata_path, exist_ok=True)
        output_file = os.path.join(metadata_path, "metadata.json")
        with open(output_file, "w") as f:
            json.dump(metadata, f, indent=2)
    elif repo_type == "ngc":
        repo_path = extract_last_part(repo_id)
        metadata_path = os.path.join(download_path, "ngc", repo_path, "metadata")
        metadata, modelmetadata = get_repo_info_ngc(repo_id, ngc_spec)
        metadataToFile = metadata["ngcMetadata"]
        os.makedirs(metadata_path, exist_ok=True)
        output_file = os.path.join(metadata_path, "metadata.yaml")
        with open(output_file, "w") as f:
            yaml.dump(
                metadataToFile,
                f,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
                Dumper=SafeUnknownDumper,
            )
        if modelmetadata:
            output_file = os.path.join(metadata_path, "modelmetadata.json")
            with open(output_file, "w") as f:
                json.dump(modelmetadata, f, indent=2)
    print("finish downloading metadata file")

    # Implement NGC repository metadata fetching if needed

    print(f"Saved metadata to {output_file}")
    return metadata


def upload_to_cloud(
    src,
    dst,
    cloud,
    token=None,
    endpoint=None,
    insecure=False,
    ca_bundle=None,
    account=None,
    container=None,
    repo_id=None,
    repo_type=None,
):
    """Upload files to cloud storage."""

    print(f"Start uploading  {repo_type} repository: {repo_id}")
    try:
        if repo_type == "ngc":
            repo_id = extract_last_part(repo_id)
        src = os.path.join(src, repo_type, repo_id)
        dst = dst + "/" + repo_type + "/" + repo_id
        if cloud == "aws":
            cmd = ["aws", "s3", "cp", src, f"{dst}/", "--recursive"]

            log_subprocess_cmd(cmd)
            subprocess.run(cmd, check=True)

        elif cloud == "azure":
            cmd = [
                "az",
                "storage",
                "blob",
                "upload-batch",
                "--account-name",
                account,
                "--destination",
                container,
                "--destination-path",
                dst,
                "--source",
                src,
            ]
            if token:
                cmd.extend(["--sas-token", token])

            subprocess.run(cmd, check=True)

        elif cloud == "pvc":
            cmd = ["aws", "s3"]

            if endpoint:
                cmd.extend(["--endpoint", endpoint])

            if insecure:
                cmd.append("--no-verify-ssl")
            elif ca_bundle:
                cmd.extend(["--ca-bundle", ca_bundle])

            cmd.extend(["cp", src, dst, "--recursive"])

            subprocess.run(cmd, check=True)

        else:
            print(f"Unsupported cloud provider: {cloud}")
            return False

        print(f"Uploaded: {src} -> {dst}")
        print(f"Finish uploading  {repo_type} repository: {repo_id} to {cloud}")
        return True

    except subprocess.SubprocessError as e:
        print(f"Error during upload: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Hugging Face model management script")

    # Help option

    # Download options
    parser.add_argument("-t", "--token", default="", help="Token for authentication")
    parser.add_argument("-j", "--json", action="store_true", help="Output raw JSON")
    parser.add_argument(
        "-do", "--download", action="store_true", help="Download the model repository"
    )
    parser.add_argument(
        "-p", "--path", default="./models", help="Path to download model files"
    )
    parser.add_argument("-ri", "--repo-id", help="Repository ID to download")
    parser.add_argument(
        "--model-name",
        dest="repo_id",
        help="Model name (alias for --repo-id; used with --repo-type mistral)",
    )
    parser.add_argument(
        "-rt", "--repo-type", default="hf", help="Repository type (default: hf)"
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose logging (DEBUG) to stderr",
    )
    license_group = parser.add_mutually_exclusive_group()
    license_group.add_argument(
        "--license-key",
        help="Rejected. Pass the Mistral license via --license-key-env",
    )
    license_group.add_argument(
        "--license-key-env",
        help="Environment variable containing the Mistral license key",
    )
    parser.add_argument(
        "--draft-model",
        help=(
            "Licensed-bucket draft/eagle weights name or weights/... path "
            "(speculative decoding). Required unless --recipe sets "
            "speculative_model_name. Prefer --recipe."
        ),
    )
    parser.add_argument(
        "--recipe",
        help=(
            "MAIS Mistral recipe YAML (model_name, weights_path, "
            "speculative_model_name); sets --repo-id when omitted"
        ),
    )
    parser.add_argument(
        "--main-weights-path",
        help=(
            "Override licensed-bucket main weights prefix (e.g. "
            "weights/mistral-small-2603); defaults from recipe or repo-id"
        ),
    )
    parser.add_argument(
        "--draft-weights-path",
        help="Override licensed-bucket draft/eagle weights prefix",
    )
    parser.add_argument(
        "--list-bucket-weights",
        action="store_true",
        help=(
            "List weights/ subdirectories visible with the Mistral license key "
            "(requires --license-key-env)"
        ),
    )

    # Upload options
    parser.add_argument(
        "-c", "--cloud", default="aws", help="Cloud provider (aws, azure, pvc)"
    )
    parser.add_argument("-s", "--src", help="Source directory for upload")
    parser.add_argument("-d", "--dst", help="Destination path in object storage")
    # TODO: remove once no external script passes --recursive.
    # Kept so existing invocations don't fail; uploads are always recursive.
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Deprecated no-op: uploads are always recursive",
    )
    parser.add_argument(
        "-e", "--endpoint", help="S3 gateway endpoint (Private cloud only)"
    )
    parser.add_argument(
        "-i", "--insecure", action="store_true", help="Allow insecure SSL connections"
    )
    parser.add_argument("-ca", "--ca-bundle", help="Path to custom CA bundle file")
    parser.add_argument("-ac", "--account", help="Account for Azure uploads")
    parser.add_argument("-cn", "--container", help="Container name for Azure uploads")
    parser.add_argument("-ns", "--ngc-spec", help="NGC spec file for downloading")

    args = parser.parse_args()
    configure_logging(verbose=args.verbose)

    # # Show help if requested or no arguments provided
    # if args.help or len(sys.argv) == 1:
    #     show_help()

    # Check requirements
    if not check_requirements(
        args.download, args.cloud if args.src else None, args.repo_type
    ):
        sys.exit(1)

    if args.list_bucket_weights:
        try:
            license_key = resolve_license_key(args)
            creds = decode_mistral_license(license_key)
            logging.info("Listing weights/ in licensed bucket %s", creds["bucket"])
            dirs = list_mistral_bucket_weight_dirs(creds)
        except (ValueError, RuntimeError) as e:
            logging.error("%s", redact_log_text(str(e)))
            sys.exit(1)
        print(f"weights/ directories in bucket {creds['bucket']}:")
        for name in dirs:
            print(f"  {name}")
        if not dirs:
            print("  (none listed — check license or rclone access)")
        sys.exit(0)

    # Handle download use case
    if args.download:
        if not args.repo_id and not args.recipe:
            print(
                "Error: --repo-id (or --model-name) or --recipe is required for download"
            )
            sys.exit(1)

        if not validate_repo_type(args.repo_type):
            sys.exit(1)

        if args.repo_type == "mistral":
            main_weights_path = None
            draft_weights_path = None
            try:
                if args.recipe:
                    recipe = load_mistral_recipe(args.recipe)
                    if not args.repo_id:
                        args.repo_id = recipe["model_name"]
                    main_weights_path = recipe["weights_path"]
                    draft_weights_path = recipe["draft_weights_path"]
                license_key = resolve_license_key(args)
                creds = decode_mistral_license(license_key)
                if args.draft_model:
                    draft_weights_path = mistral_bucket_weights_path(args.draft_model)
                if main_weights_path is None:
                    main_weights_path = mistral_bucket_weights_path(args.repo_id)
                if args.main_weights_path:
                    main_weights_path = mistral_bucket_weights_path(
                        args.main_weights_path
                    )
                if args.draft_weights_path:
                    draft_weights_path = mistral_bucket_weights_path(
                        args.draft_weights_path
                    )
                require_mistral_draft_weights(draft_weights_path, creds)
                logging.info(
                    "Resolved Mistral download: repo_id=%r main_weights=%r draft_weights=%r",
                    args.repo_id,
                    main_weights_path,
                    draft_weights_path,
                )
            except (ValueError, FileNotFoundError) as e:
                logging.error("%s", redact_log_text(str(e)))
                sys.exit(1)
            if not download_repo(
                args.repo_id,
                args.token,
                args.path,
                args.repo_type,
                None,
                draft=draft_weights_path,
                creds=creds,
                main_weights_path=main_weights_path,
            ):
                sys.exit(1)
            sys.exit(0)

        # Get repository info and download
        metadata = get_repo_info(
            args.repo_id, args.token, args.repo_type, args.path, args.ngc_spec
        )
        if metadata is None:
            print("Error: Failed to get repository metadata")
            sys.exit(1)
        if not download_repo(
            args.repo_id, args.token, args.path, args.repo_type, metadata
        ):
            sys.exit(1)
        sys.exit(0)

    # Handle upload use case
    if args.src and args.dst:
        if not upload_to_cloud(
            args.src,
            args.dst,
            args.cloud,
            args.token,
            args.endpoint,
            args.insecure,
            args.ca_bundle,
            args.account,
            args.container,
            args.repo_id,
            args.repo_type,
        ):
            sys.exit(1)
        print("Upload completed.")
    else:
        print("Error: Missing required parameters.")
        show_help()

# argv flags whose following value (or =value) must not appear in logs
_SENSITIVE_CMD_FLAGS = frozenset(
    {
        "--token",
        "--sas-token",
        "--license-key",
        "--access-key-id",
        "--secret-access-key",
        "--sharedaccesskey",
    }
)

_LOG_REDACTIONS = (
    (
        re.compile(r'(?i)(access_key_id["\']?\s*[:=]\s*["\']?)[^"\'\s,}]+'),
        r"\1<redacted>",
    ),
    (
        re.compile(r'(?i)(secret_access_key["\']?\s*[:=]\s*["\']?)[^"\'\s,}]+'),
        r"\1<redacted>",
    ),
    (re.compile(r"nvapi-[A-Za-z0-9_-]+"), "nvapi-<redacted>"),
    (re.compile(r"(?i)(Bearer\s+)[A-Za-z0-9._-]+"), r"\1<redacted>"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "AKIA<redacted>"),
    (re.compile(r"(?i)(--token[=\s])\S+"), r"\1<redacted>"),
    (re.compile(r"(?i)(--sas-token[=\s])\S+"), r"\1<redacted>"),
)


def redact_cmd_argv(cmd):
    """Return a copy of argv safe to log (tokens and keys replaced)."""
    if not cmd:
        return []
    argv = list(cmd)
    safe = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg.startswith("NGC_API_KEY="):
            safe.append("NGC_API_KEY=<redacted>")
            i += 1
            continue
        if "=" in arg:
            name, _, _value = arg.partition("=")
            if name in _SENSITIVE_CMD_FLAGS or name.lower() in _SENSITIVE_CMD_FLAGS:
                safe.append(f"{name}=<redacted>")
                i += 1
                continue
        if arg in _SENSITIVE_CMD_FLAGS:
            safe.append(arg)
            if i + 1 < len(argv):
                safe.append("<redacted>")
                i += 2
                continue
        safe.append(arg)
        i += 1
    return safe


def redact_log_text(text):
    """Scrub secrets from subprocess or error text before logging."""
    if not text:
        return ""
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="replace")
    for pattern, repl in _LOG_REDACTIONS:
        text = pattern.sub(repl, text)
    return text


def configure_logging(verbose=False):
    """Send script progress to stderr so stdout stays free for piping."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def log_subprocess_cmd(cmd):
    """Log an external command without secrets (argv only)."""
    logging.info("Running command: %s", " ".join(redact_cmd_argv(cmd)))


def log_subprocess_output(level, label, text):
    """Log subprocess stdout/stderr after redacting credentials."""
    logging.log(level, "%s: %s", label, redact_log_text(text))



if __name__ == "__main__":
    main()
