#!/usr/bin/env python3

import argparse
import json
import os
import base64
import subprocess
import sys
import logging
from pathlib import Path
from datetime import datetime

import yaml
from typing import List, Dict, Optional, Tuple

# courtesy : https://stackoverflow.com/questions/43765849/pyyaml-load-and-dump-yaml-file-and-preserve-tags-customtag
class SafeUnknownConstructor(yaml.constructor.SafeConstructor):
    def __init__(self):
        yaml.constructor.SafeConstructor.__init__(self)

    def construct_undefined(self, node):
        data = getattr(self, 'construct_' + node.id)(node)
        datatype = type(data)
        wraptype = type('TagWrap_'+datatype.__name__, (datatype,), {})
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
        if type(wrapdata).__name__.startswith('TagWrap_'):
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

    def __init__(self, stream,
            default_style=None, default_flow_style=False,
            canonical=None, indent=None, width=None,
            allow_unicode=None, line_break=None,
            encoding=None, explicit_start=None, explicit_end=None,
            version=None, tags=None, sort_keys=True):

        SafeUnknownRepresenter.__init__(self, default_style=default_style,
                default_flow_style=default_flow_style, sort_keys=sort_keys)

        yaml.dumper.SafeDumper.__init__(self,  stream,
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
                                        sort_keys=sort_keys)


MySafeLoader = SafeUnknownLoader
yaml.constructor.SafeConstructor.add_constructor(None, SafeUnknownConstructor.construct_undefined)

class ModelParser:
    def __init__(self, yaml_file_path: str):
        """Initialize the parser with a YAML file path."""
        self.yaml_file_path = yaml_file_path
        self.models_data = self._load_yaml()
    
    def _load_yaml(self) -> Dict:
        """Load and parse the YAML file."""
        try:
            with open(self.yaml_file_path, "r") as file:
                yaml_data = file.read()
                return yaml.load(yaml_data,Loader=MySafeLoader)
        except FileNotFoundError:
            print(f"Error: File {self.yaml_file_path} not found.")
            return {}
        except yaml.YAMLError as e:
            print(f"Error parsing YAML: {e}")
            return {}
    
    def get_all_models(self) -> List[Dict[str, str]]:
        """Fetch all models with their basic information."""
        models = []
        if 'models' in self.models_data:
            for model in self.models_data['models']:
                model_info = {
                    'name': model.get('name', ''),
                    'displayName': model.get('displayName', ''),
                    'modelHubID': model.get('modelHubID', ''),
                    'category': model.get('category', ''),
                    'description': model.get('description', '')
                }
                models.append(model_info)
        return models
    
    def get_model_variant_ids(self, model_name: str) -> List[str]:
        """
        Fetch all model variant IDs for a given model name.
        
        Args:
            model_name: The name of the model to search for
            
        Returns:
            List of variant IDs for the specified model
        """
        variant_ids = []
        if 'models' in self.models_data:
            for model in self.models_data['models']:
                if model.get('name', '').lower() == model_name.lower():
                    if 'modelVariants' in model:
                        for variant in model['modelVariants']:
                            variant_id = variant.get('variantId', '')
                            if variant_id:
                                variant_ids.append(variant_id)
                    break
        return variant_ids
    
    def get_optimization_profile_ids(self, model_name: str, variant_id: str = None) -> List[str]:
        """
        Fetch all optimization profile IDs for a given model name and optional variant ID.
        
        Args:
            model_name: The name of the model
            variant_id: Optional variant ID. If None, gets profiles from all variants
            
        Returns:
            List of optimization profile IDs
        """
        profile_ids = []
        if 'models' in self.models_data:
            for model in self.models_data['models']:
                if model.get('name', '').lower() == model_name.lower():
                    if 'modelVariants' in model:
                        for variant in model['modelVariants']:
                            # If variant_id is specified, only process that variant
                            if variant_id and variant.get('variantId', '').lower() != variant_id.lower():
                                continue
                            
                            if 'optimizationProfiles' in variant:
                                for profile in variant['optimizationProfiles']:
                                    profile_id = profile.get('profileId', '')
                                    if profile_id:
                                        profile_ids.append(profile_id)
                    break
        return profile_ids
    
    def get_detailed_model_info(self, model_name: str) -> Optional[Dict]:
        """
        Get detailed information about a specific model including all variants and profiles.
        
        Args:
            model_name: The name of the model
            
        Returns:
            Dictionary containing detailed model information
        """
        if 'models' in self.models_data:
            for model in self.models_data['models']:
                if model.get('name', '').lower() == model_name.lower():
                    return model
        return None
    



def extract_last_part(input_string):
    # Split the string by '/' and return the last part
    return input_string.split('/')[-1]

def run_ngc_info_command(ID):
    logging.info(f"Start Fetching model info from NGC: {ID}")
    cmd = ["ngc", "registry", "model", "info", ID, "--format_type", "json"]
    
    # print the command without exposing Cloudera's key
    logging.info(cmd)
    
    # Set up environment with API keys
    # env = os.environ.copy()
    # env["NGC_CLI_API_KEY"] = os.environ.get("NGC_API_KEY")
    # env["NGC_CLI_ORG"] = os.environ.get("NGC_CLI_ORG")
    
    # Run the command and collect output
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=False,
            check=True
        )
        
        logging.info(f"Finish Fetching NGC model repo {ID}")
        
        # Parse the JSON output
        try:
            metadata_map = json.loads(result.stdout)
            return metadata_map, None
        except json.JSONDecodeError as e:
            logging.error(f"Error while Unmarshalling the NGC info command to map: {str(e)}")
            return None, e
            
    except subprocess.CalledProcessError as e:
        logging.error(f"Error Fetching model repo: {ID}")
        logging.error(f"Error: {str(e)}")
        return None, e



def get_ngc_model_info(model_id, tag):
    model_metadata_map, err = run_ngc_info_command(model_id)
    if err:
        logging.error(f"Error Fetching model repo: {model_id}")
        logging.error(f"Error: {str(err)}")
        return ""
    versionMetadataMap, err = run_ngc_info_command(model_id + ":" + tag)
    if err:
        logging.error(f"Error Fetching model repo: {model_id}")
        logging.error(f"Error: {str(err)}")
        return ""
    v1 = versionMetadataMap.get("totalSizeInBytes", "")
    v2 = model_metadata_map.get("versionId", "")
    if v2 != "":
        model_metadata_map["versionId"] = v2
    return model_metadata_map


def load_ngc_spec(spec_file):
    with open(spec_file, "r") as file:
        yaml_data = file.read()

    yaml_data = yaml.load(yaml_data,Loader=MySafeLoader)
    return yaml_data


def execute_nim_download_command(repo_id, folder_location, ngc_spec, profile_sha, version):
    """
    Execute nim cli download command to download model files.
    
    Args:
        repo_id (str): Repository ID
    """
    model_name = repo_id.split(":")[0]
    count = model_name.count('/')
    if count != 2:
        raise ValueError(f"Expected 3 '/' characters, but found {count} in model name")

    # Get the absolute path of the ngc_spec folder
    ngc_spec_abs = os.path.dirname(ngc_spec)
    manifest_path = f"{ngc_spec_abs}/manifests/{version}/{model_name}.yaml"
    cmd = [
        "nimcli", "download", "--profiles", profile_sha, "--manifest-file",
        manifest_path, "--model-cache-path", folder_location
    ]

    print(cmd)
        
    try:
        # output = subprocess.check_output(cmd, env=env, stderr=subprocess.STDOUT)
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        logging.error(f"Error download model repo: {repo_id}")
        logging.error(f"Error: {str(e)}")
        return folder_location, e


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

    if files and len(files) > 0:
        # Download specific files
        for file in files:
            cmd = [
                "ngc", "registry", "model", "download-version", 
                repo_id, "--file", file, "--dest", folder_location, 
                "--format_type", "json"
            ]
            logging.info(cmd)
            
            try:
                # output = subprocess.check_output(cmd,env=env, stderr=subprocess.STDOUT)
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                logging.error(f"Error download model repo: {repo_id}")
                logging.error(f"Error: {str(e)}")
                logging.error(f"Command output: {e.output.decode()}")
                return folder_location, e
    else:
        # Download entire repository
        cmd = [
            "ngc", "registry", "model", "download-version", 
            repo_id, "--dest", folder_location, "--format_type", "json"
        ]
        logging.info(cmd)
        
        try:
            # output = subprocess.check_output(cmd, env=env, stderr=subprocess.STDOUT)
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            logging.error(f"Error download model repo: {repo_id}")
            logging.error(f"Error: {str(e)}")
            logging.error(f"Command output: {e.output.decode()}")
            return folder_location, Exception(e.output.decode())
    
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
        'found': False,
        'components': [],
        'ngcMetadata': None,
    }
    modelCard = ''
    # Iterate through models
    for model in data.get('models', []):
        # Iterate through variants
        for variant in model.get('modelVariants', []):
            modelCard = variant.get('modelCard', '')
            # Iterate through profiles
            for profile in variant.get('optimizationProfiles', []):
                # Check if this is the target profileID
                if profile.get('profileId') == target_profile_id:
                    result['found'] = True
                    result['ngcMetadata'] = profile.get('ngcMetadata', None)
                    # Look for ngcMetadata which contains the workspace components
                    for sha_key, metadata in profile.get('ngcMetadata', {}).items():
                        if 'workspace' in metadata and 'components' in metadata['workspace']:
                            for component in metadata['workspace']['components']:
                                component_info = {
                                    'destination': component.get('dst', ''),
                                    'repo_id': component.get('src', {}).get('repo_id', '')
                                }
                                
                                # Extract files if they exist
                                files = component.get('src', {}).get('files', [])
                                if files:
                                    # Handle both direct strings and dictionaries with name tags
                                    component_info['files'] = []
                                    for f in files:
                                        if isinstance(f, str):
                                            component_info['files'].append(f)
                                        elif isinstance(f, dict) and f.get('!name'):
                                            component_info['files'].append(f.get('!name'))
                                
                                result['components'].append(component_info)
                    
                    return result, modelCard  # Return once the profile is found
    
    return result, modelCard  # Return not found if we get here


def show_help():
    """Display help information and exit."""
    help_text = """
    Description: Fetches information about a model from the Hugging Face Hub and optionally downloads it,
    or uploads model artifacts to cloud storage.
    Commands:
      configure           Run interactive configuration setup to save default values
      list-local-models   List all downloaded models with their download status and retry count
    Download Options:
      -do, --download         Download the model repository
      -ri, --repo-id          Repository ID to download (required with --download)
      -rt, --repo-type        Repository type: hf (HuggingFace) or ngc (NVIDIA GPU Cloud) [default: hf]
      -p, --path              Path to download model files [default: from config or ~/.airgap/model]
      -t, --token             Token for authentication
      -ns, --ngc-spec         NGC spec file for downloading (required for NGC downloads)
    Upload Options:
      -c, --cloud             Cloud provider: aws, azure, or pvc
      -s, --src               Source directory for upload
      -d, --dst               Destination path in object storage
      -r, --recursive         Recursively upload folders
      -e, --endpoint          S3 gateway endpoint (for pvc only)
      -i, --insecure          Allow insecure SSL connections
      -ca, --ca-bundle        Path to custom CA bundle file
      -ac, --account          Account for Azure uploads
      -cn, --container        Container name for Azure uploads
    Examples:
      # Configure default settings
      python import_to_airgap.py configure
      # List all downloaded models
      python import_to_airgap.py list-local-models
      # Download a HuggingFace model
      python import_to_airgap.py -do --repo-id facebook/bart-large --repo-type hf
      # Download an NGC model
      python import_to_airgap.py -do --repo-id nim/meta/llama-3.1-8b-instruct:1.0.0 --repo-type ngc --ngc-spec spec.yaml
      # Upload to cloud storage wih configured defaults
      python import_to_airgap.py --repo-id bert-base-uncased --repo-type hf
      # Upload to cloud storage with overridden defaults
      # Upload to AWS S3
      python import_to_airgap.py --cloud aws --src /path/to/models/ --dst s3://bucket/path --repo-id bert-base-uncased --repo-type hf
      # Upload to private cloud with S3 endpoint
      python import_to_airgap.py --cloud pvc --src /path/to/model/hf/meta/llama3.1 --dst s3://bucket/secured-models/hf/meta/llama3.1 --repo-id meta/llama3.1 --repo-type hf
      # Upload to Azure Blob Storage
      python import_to_airgap.py --cloud azure --src /path/to/model --account cloudera-customer1 --container data --dst modelregistry/secured-models/hf/meta/llama3.1 --repo-id meta/llama3.1 --repo-type hf
    """
    print(help_text)
    sys.exit(1)


def check_requirements(download_model, cloud):
    """Check if the required tools are installed."""
    if download_model:
        try:
            subprocess.run(["hf", "version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except (subprocess.SubprocessError, FileNotFoundError):
            print("Error: huggingface-cli is required for downloading but not installed.")
            print("Please install it using pip:")
            print("pip install --upgrade huggingface_hub")
            return False
        try:
            subprocess.run(["ngc", "version", "info"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except (subprocess.SubprocessError, FileNotFoundError):
            print("Error: NGC CLI is required for downloading but not installed.")
            print("Please install it ngc cli : https://org.ngc.nvidia.com/setup/installers/cli")
            return False
    if cloud:
        if cloud == "aws" or cloud == "pvc":
            try:
                subprocess.run(["aws", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except (subprocess.SubprocessError, FileNotFoundError):
                print(f"Error: aws-cli is required for uploading to {cloud} but not installed.")
                print("Please install it using pip:")
                print("  pip install awscli")
                return False
        elif cloud == "azure":
            try:
                subprocess.run(["az", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except (subprocess.SubprocessError, FileNotFoundError):
                print("Error: Azure CLI is required for uploading to Azure Blob Storage but not installed.")
                print("Please install it using pip:")
                print("  pip install azure-cli")
                return False
        else:
            print(f"Unsupported cloud provider: {cloud}")
            return False
    
    return True

def validate_repo_type(repo_type):
    """Validate the repository type."""
    valid_types = ["hf", "ngc"]
    if repo_type not in valid_types:
        print(f"Error: Invalid repository type '{repo_type}'")
        print(f"Valid types are: {', '.join(valid_types)}")
        return False
    return True

def download_repo_hf(repo_id, token, download_path):
    """Download a Hugging Face repository using huggingface-cli."""

    print(f"Downloading repository: {repo_id} to {download_path}")
    
    # Create necessary directories
    os.makedirs(os.path.join(download_path, "hf", repo_id, "artifacts"), exist_ok=True)
    
    # Download the repository
    print("Starting download with huggingface-cli...")
    try:
        cmd = ["hf", "download", repo_id, "--local-dir", f"{download_path}/hf/{repo_id}/artifacts"]
        if token:
            cmd.extend(["--token", token])
        
        subprocess.run(cmd, check=True)
        print(f"Download completed successfully for {repo_id}")
        return True
    except subprocess.SubprocessError:
        print(f"Error: Failed to download {repo_id}")
        return False


def canusenimcli(metadata):
    """
    Parse YAML string and extract release information.
    
    Args:
        yaml_str (str): YAML string containing release data
        
    Returns:
        dict: Dictionary containing parsed release information
    """
    try:
        # Parse the YAML string

        
        # Extract information from the first entry in the YAML
        # (In this case, there's only one entry with a hash as the key)
        ngcmetadata = metadata['ngcMetadata']
        
        print(ngcmetadata)
        hash_key = next(iter(ngcmetadata))
        release_info = ngcmetadata[hash_key]
        print(release_info)
        # Check if the release key is present
        if "release" not in release_info:
            return False
        if "model" in release_info:
            if release_info['model'] ==  "nvidia/nemoretriever-parse":
                return True
        if "release" in release_info:
            version = release_info["release"]
            checkversion = "1.3.0"
            from packaging.version import Version
            return Version(version) >= Version(checkversion)
    except Exception as e:
        print("error Failed to parse YAML:", e)
        return False

def download_repo(repo_id, token, download_path, repo_type, metadata, ngc_spec):
    """Download a repository using huggingface-cli."""
    print(f"start downloading {repo_type} repository: {repo_id}")
    if repo_type == 'hf':
        return download_repo_hf(repo_id, token, download_path)
    elif repo_type == 'ngc':
        ngcPrefix = 'ngc://'
        if repo_id.startswith(ngcPrefix):
            repo_id = repo_id[len(ngcPrefix):]
        repo_path=extract_last_part(repo_id)
        download_path = os.path.join(download_path, "ngc", repo_path, "artifacts")
        os.makedirs(download_path, exist_ok=True)
        # Implement NGC repository downloading if needed
        nimcli=canusenimcli(metadata)
        ngcmetadata = metadata['ngcMetadata']
        profile_sha = next(iter(ngcmetadata))
        spec = load_ngc_spec(ngc_spec)
        version = ''
        if 'registryVersion' in spec:
            version = spec['registryVersion']
        if nimcli:
            model_name = repo_id.split(":")[0]
            model_name_parts = model_name.split("/")
            if len(model_name_parts)!=3:
                raise NameError("Model name should have three parts "+model_name)
            execute_nim_download_command(repo_id, download_path, ngc_spec, profile_sha, version)
        else:
            for component in metadata['components']:
                print(f"Repo ID: {component['repo_id']}")
                print(f"Destination: {component['destination']}")
                execute_ngc_download_command(component['repo_id'], download_path, component.get('files'))
    print("Finish downloading artifacts")


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
    ngcMetadata, modelCard =  extract_profile_components(spec, repo_id)
    repo_id=repo_id.split(':')
    try:
        modelMetadata = json.loads(base64.b64decode(modelCard).decode('utf-8'))
    except Exception:
        modelMetadata = {}
    return ngcMetadata, modelMetadata


def get_repo_info(repo_id, token, repo_type, download_path, ngc_spec):
    """Get repository metadata and save it to a file."""
    print(f"Fetching information for {repo_type} repository: {repo_id}")
    print(f"Download path: {download_path}")
    if repo_type == 'hf':
        metadata_path = os.path.join(download_path, "hf", repo_id, "metadata")
        metadata = get_repo_info_hf(repo_id, token)
        os.makedirs(metadata_path, exist_ok=True)
        output_file = os.path.join(metadata_path, "metadata.json")
        with open(output_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    elif repo_type == 'ngc':
        repo_path = extract_last_part(repo_id)
        metadata_path = os.path.join(download_path, "ngc", repo_path, "metadata")
        metadata, modelmetadata =  get_repo_info_ngc(repo_id, ngc_spec)
        metadataToFile = metadata['ngcMetadata']
        os.makedirs(metadata_path, exist_ok=True)
        output_file = os.path.join(metadata_path, "metadata.yaml")
        with open(output_file, 'w') as f:
            yaml.dump(metadataToFile, f, default_flow_style=False, sort_keys=False, allow_unicode=True, Dumper=SafeUnknownDumper)
        if modelmetadata:
            output_file = os.path.join(metadata_path, "modelmetadata.json")
            with open(output_file, 'w') as f:
                json.dump(modelmetadata, f, indent=2)
    print("finish downloading metadata file")

        # Implement NGC repository metadata fetching if needed

        
    print(f"Saved metadata to {output_file}")
    return metadata


def upload_to_cloud(src, dst, cloud, token=None, recursive=False, endpoint=None, 
                    insecure=False, ca_bundle=None, account=None, container=None,
                    repo_id=None, repo_type=None):
    """Upload files to cloud storage."""

    print(f"Start uploading  {repo_type} repository: {repo_id}")
    try:
        if repo_type == "ngc":
            repo_id = extract_last_part(repo_id)
        src = os.path.join(src, repo_type, repo_id)
        dst = dst +"/"+ repo_type +"/"+ repo_id
        if cloud == "aws":
            cmd = ["aws", "s3", "cp", src, f"{dst}/", "--recursive"]
            
            print(" ".join(cmd))
            subprocess.run(cmd, check=True)
            
        elif cloud == "azure":
            cmd = [
                "az", "storage", "blob", "upload-batch",
                "--account-name", account,
                "--destination", container,
                "--destination-path", dst,
                "--source", src
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


def print_models(models: List[Dict[str, str]], title: str = "Models"):
    """Print models in a formatted way."""
    print(f"\n=== {title.upper()} ===")
    if not models:
        print("No models found.")
        return
    
    for i, model in enumerate(models, 1):
        print(f"{i}. {model['name']}")
        print(f"   Display Name: {model['displayName']}")
        print(f"   Category: {model['category']}")
        print(f"   Hub ID: {model['modelHubID']}")
        if model['description']:
            desc = model['description'][:100] + "..." if len(model['description']) > 100 else model['description']
            print(f"   Description: {desc}")
        print()

def print_list(items: List[str], title: str):
    """Print a list of items in a formatted way."""
    print(f"\n=== {title.upper()} ===")
    if not items:
        print(f"No {title.lower()} found.")
        return
    
    for i, item in enumerate(items, 1):
        print(f"{i}. {item}")

def print_detailed_model(model_data: Dict, model_name: str):
    """Print detailed model information."""
    print(f"\n=== DETAILED INFO FOR '{model_name}' ===")
    
    if not model_data:
        print(f"Model '{model_name}' not found.")
        return
    
    print(f"Name: {model_data.get('name', 'N/A')}")
    print(f"Display Name: {model_data.get('displayName', 'N/A')}")
    print(f"Model Hub ID: {model_data.get('modelHubID', 'N/A')}")
    print(f"Category: {model_data.get('category', 'N/A')}")
    print(f"Type: {model_data.get('type', 'N/A')}")
    print(f"Description: {model_data.get('description', 'N/A')}")
    print(f"License: {model_data.get('license', 'N/A')}")
    
    if 'labels' in model_data:
        print(f"Labels: {', '.join(model_data['labels'])}")
    
    if 'modelVariants' in model_data:
        print(f"\nModel Variants ({len(model_data['modelVariants'])}):")
        for i, variant in enumerate(model_data['modelVariants'], 1):
            print(f"  {i}. {variant.get('variantId', 'N/A')}")
            if 'optimizationProfiles' in variant:
                print(f"     Optimization Profiles ({len(variant['optimizationProfiles'])}):")
                for j, profile in enumerate(variant['optimizationProfiles'], 1):
                    profile_id = profile.get('profileId', 'N/A')
                    display_name = profile.get('displayName', 'N/A')
                    framework = profile.get('framework', 'N/A')
                    print(f"       {j}. {profile_id}")
                    print(f"          Display Name: {display_name}")
                    print(f"          Framework: {framework}")


def get_config_path():
    """Get the path to the configuration file."""
    home = Path.home()
    config_dir = home / ".airgap"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir / "config.json"


def get_metadata_file_path(download_path):
    """Get the path to the models-metadata.yaml file."""
    metadata_file = os.path.join(download_path, "models-metadata.yaml")
    return metadata_file


def load_models_metadata(download_path):
    """Load models metadata from YAML file."""
    metadata_file = get_metadata_file_path(download_path)
    if os.path.exists(metadata_file):
        try:
            with open(metadata_file, 'r') as f:
                data = yaml.safe_load(f)
                return data if data else []
        except (yaml.YAMLError, IOError) as e:
            print(f"Warning: Could not load metadata file: {e}")
            return []
    return []


def save_models_metadata(download_path, metadata):
    """Save models metadata to YAML file."""
    metadata_file = get_metadata_file_path(download_path)
    try:
        with open(metadata_file, 'w') as f:
            yaml.dump(metadata, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    except IOError as e:
        print(f"Error: Could not save metadata file: {e}")


def update_model_metadata(download_path, repo_type, repo_id, download_success, start_time, finish_time):
    """Update or add model metadata entry."""
    metadata = load_models_metadata(download_path)
    
    # Find existing entry for this repo-type and repo-id
    existing_entry = None
    for entry in metadata:
        if entry.get('repo-type') == repo_type and entry.get('repo-id') == repo_id:
            existing_entry = entry
            break
    
    if existing_entry:
        # Update existing entry
        existing_entry['num-retries'] = existing_entry.get('num-retries', 0) + 1
        existing_entry['download-starttime'] = start_time
        existing_entry['download-finishtime'] = finish_time
        existing_entry['download-success'] = download_success
    else:
        # Create new entry
        new_entry = {
            'repo-type': repo_type,
            'repo-id': repo_id,
            'download-starttime': start_time,
            'download-finishtime': finish_time,
            'download-success': download_success,
            'num-retries': 0
        }
        metadata.append(new_entry)
    
    save_models_metadata(download_path, metadata)
    print(f"Metadata updated for {repo_type}/{repo_id}")


def list_local_models(download_path):
    """List all models from the models-metadata.yaml file."""
    metadata = load_models_metadata(download_path)
    
    if not metadata:
        print("No models found in metadata.")
        return
    
    print("\n=== LOCAL MODELS ===")
    print(f"{'Repo Type':<15} {'Repo ID':<50} {'Download Success':<20} {'Num Retries':<15}")
    print("=" * 100)
    
    for entry in metadata:
        repo_type = entry.get('repo-type', 'N/A')
        repo_id = entry.get('repo-id', 'N/A')
        download_success = 'true' if entry.get('download-success', False) else 'false'
        num_retries = entry.get('num-retries', 0)
        
        print(f"{repo_type:<15} {repo_id:<50} {download_success:<20} {num_retries:<15}")
    
    print()


def load_config():
    """Load configuration from file."""
    config_path = get_config_path()
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load config file: {e}")
            return {}
    return {}


def save_config(config):
    """Save configuration to file."""
    config_path = get_config_path()
    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"\nConfiguration saved to {config_path}")
    except IOError as e:
        print(f"Error: Could not save config file: {e}")
        sys.exit(1)


def configure_interactive():
    """Interactively configure the tool settings."""
    print("=== Airgap Model Import Configuration ===\n")
    
    config = load_config()
    
    # Get default model path
    default_path = str(Path.home() / ".airgap" / "model")
    current_path = config.get('path', default_path)
    path_input = input(f"Enter default model download/source path [{current_path}]: ").strip()
    config['path'] = path_input if path_input else current_path
    
    # Get authentication token
    current_token = config.get('token', '')
    token_display = '*' * 8 if current_token else 'none'
    token_input = input(f"Enter authentication token (for HuggingFace) [current: {token_display}]: ").strip()
    if token_input:
        config['token'] = token_input
    elif not current_token:
        config['token'] = ''
    
    # Get cloud provider
    current_cloud = config.get('cloud', 'aws')
    cloud_input = input(f"Enter cloud provider (aws/azure/pvc) [{current_cloud}]: ").strip().lower()
    config['cloud'] = cloud_input if cloud_input else current_cloud
    
    # Validate cloud provider
    if config['cloud'] not in ['aws', 'azure', 'pvc']:
        print(f"Warning: Invalid cloud provider '{config['cloud']}'. Defaulting to 'aws'.")
        config['cloud'] = 'aws'
    
    # Get destination path
    current_dst = config.get('dst', '')
    dst_input = input(f"Enter default destination path in cloud storage [{current_dst or 'none'}]: ").strip()
    config['dst'] = dst_input if dst_input else current_dst
    
    # Cloud-specific configuration
    if config['cloud'] in ['aws', 'pvc']:
        print(f"\n--- AWS/PVC Configuration ---")
        
        if config['cloud'] == 'pvc':
            current_endpoint = config.get('endpoint', '')
            endpoint_input = input(f"Enter S3 endpoint URL [{current_endpoint or 'none'}]: ").strip()
            config['endpoint'] = endpoint_input if endpoint_input else current_endpoint
            
            current_insecure = config.get('insecure', False)
            insecure_input = input(f"Allow insecure SSL connections? (yes/no) [{'yes' if current_insecure else 'no'}]: ").strip().lower()
            if insecure_input in ['yes', 'y']:
                config['insecure'] = True
                config.pop('ca_bundle', None)  # Remove ca_bundle if insecure is enabled
            elif insecure_input in ['no', 'n']:
                config['insecure'] = False
                current_ca = config.get('ca_bundle', '')
                ca_input = input(f"Enter path to CA bundle file [{current_ca or 'none'}]: ").strip()
                config['ca_bundle'] = ca_input if ca_input else current_ca
            # If no input, keep current settings
        else:
            # For AWS, clear PVC-specific settings
            config.pop('endpoint', None)
            config.pop('insecure', None)
            config.pop('ca_bundle', None)
        
        # Clear Azure-specific settings
        config.pop('account', None)
        config.pop('container', None)
    
    elif config['cloud'] == 'azure':
        print(f"\n--- Azure Configuration ---")
        
        current_account = config.get('account', '')
        account_input = input(f"Enter Azure storage account name [{current_account or 'none'}]: ").strip()
        config['account'] = account_input if account_input else current_account
        
        current_container = config.get('container', '')
        container_input = input(f"Enter Azure container name [{current_container or 'none'}]: ").strip()
        config['container'] = container_input if container_input else current_container
        
        # Clear AWS/PVC-specific settings
        config.pop('endpoint', None)
        config.pop('insecure', None)
        config.pop('ca_bundle', None)
    
    # Save configuration
    save_config(config)
    print("\nConfiguration complete!")
    return config


def main():
    # Load configuration first to get defaults
    config = load_config()
    
    # Set up default values from config
    default_path = config.get('path', str(Path.home() / ".airgap" / "model"))
    default_token = config.get('token', '')
    default_cloud = config.get('cloud', 'aws')
    default_dst = config.get('dst', '')
    default_endpoint = config.get('endpoint', '')
    default_ca_bundle = config.get('ca_bundle', '')
    default_account = config.get('account', '')
    default_container = config.get('container', '')
    
    parser = argparse.ArgumentParser(description="Airgap model import and management tool", add_help=False)
    parser.add_argument("-h", "--help", action="store_true", help="Show this help message and exit")
    
    # Configuration option
    parser.add_argument("--configure", action="store_true", help="Configure default settings interactively")
    
    # Download options
    parser.add_argument("-t", "--token", default=None, help=f"Token for authentication (default: configured value)")
    parser.add_argument("-j", "--json", action="store_true", help="Output raw JSON")
    parser.add_argument("-do", "--download", action="store_true", help="Download the model repository")
    parser.add_argument("-p", "--path", default=None, help=f"Path to download/source model files (default: {default_path})")
    parser.add_argument("-ri", "--repo-id", help="Repository ID to download")
    parser.add_argument("-rt", "--repo-type", default="hf", help="Repository type (default: hf)")
    
    # Upload options
    parser.add_argument("-c", "--cloud", default=None, help=f"Cloud provider (aws, azure, pvc) (default: {default_cloud})")
    parser.add_argument("-s", "--src", default=None, help=f"Source directory for upload (default: same as --path)")
    parser.add_argument("-d", "--dst", default=None, help=f"Destination path in object storage (default: configured value)")
    parser.add_argument("-r", "--recursive", action="store_true", help="Recursively upload folders")
    parser.add_argument("-e", "--endpoint", default=None, help=f"S3 gateway endpoint (Private cloud only) (default: configured value)")
    parser.add_argument("-i", "--insecure", action="store_true", help="Allow insecure SSL connections")
    parser.add_argument("-ca", "--ca-bundle", default=None, help=f"Path to custom CA bundle file (default: configured value)")
    parser.add_argument("-ac", "--account", default=None, help=f"Account for Azure uploads (default: configured value)")
    parser.add_argument("-cn", "--container", default=None, help=f"Container name for Azure uploads (default: configured value)")
    parser.add_argument("-ns", "--ngc-spec", help="NGC spec folder path")
    
    # Query options
    parser.add_argument('-m', '--model-name', help='Name of the model to query')
    parser.add_argument('-vid', '--variant-id', help='Variant ID (used with model name for specific queries)')
    parser.add_argument('--list-all', action='store_true', help='List all available models')
    parser.add_argument('--list-variants', action='store_true', help='List all variant IDs for the specified model')
    parser.add_argument('--list-profiles', action='store_true', help='List all optimization profile IDs for the specified model (and variant if provided)')
    parser.add_argument('--list-local-models', action='store_true', help='List all locally downloaded models from metadata')

    # Support 'help' command by converting it to --help flag
    if len(sys.argv) > 1 and sys.argv[1] == 'help':
        sys.argv[1] = '--help'

    args = parser.parse_args()
    
    # Handle configure command
    if args.configure:
        configure_interactive()
        sys.exit(0)
    
    # Apply defaults from config if not provided in command line
    if args.token is None:
        args.token = default_token
    if args.path is None:
        args.path = default_path
    if args.cloud is None:
        args.cloud = default_cloud
    if args.src is None:
        args.src = default_path  # src defaults to path
    if args.dst is None:
        args.dst = default_dst
    if args.endpoint is None:
        args.endpoint = default_endpoint
    if args.ca_bundle is None:
        if default_ca_bundle and default_ca_bundle != '':
            args.ca_bundle = default_ca_bundle
    if args.account is None:
        if default_account and default_account != '':
            args.account = default_account
    if args.container is None:
        if default_container and default_container != '':
            args.container = default_container
    
    # Handle insecure flag - if explicitly set in command line, it overrides config
    # Otherwise, use config value
    if not args.insecure and config.get('insecure', False):
        args.insecure = True
    
    # # Show help if requested or no arguments provided
    if args.help or len(sys.argv) == 1:
        show_help()
    
    # Check requirements
    if not check_requirements(args.download, args.cloud if args.src else None):
        sys.exit(1)
    
    # Handle list-local-models command
    if args.list_local_models:
        list_local_models(args.path)
        return
    
    # Handle different command combinations
    if args.list_all:
        ngc_spec_file = args.ngc_spec
        parser = ModelParser(ngc_spec_file)
        models = parser.get_all_models()
        print_models(models, "All Models")
        return
    
    elif args.model_name:
        ngc_spec_file = args.ngc_spec
        parser = ModelParser(ngc_spec_file)
        
        if args.list_variants:
            variants = parser.get_model_variant_ids(args.model_name)
            print_list(variants, f"Variants for '{args.model_name}'")
        
        elif args.list_profiles:
            profiles = parser.get_optimization_profile_ids(args.model_name, args.variant_id)
            if args.variant_id:
                title = f"Optimization Profiles for '{args.model_name}' variant '{args.variant_id}'"
            else:
                title = f"Optimization Profiles for '{args.model_name}'"
            print_list(profiles, title)
        
        else:
            # Default: show basic info about the model
            model_data = parser.get_detailed_model_info(args.model_name)
            if model_data:
                model_info = {
                    'name': model_data.get('name', ''),
                    'displayName': model_data.get('displayName', ''),
                    'modelHubID': model_data.get('modelHubID', ''),
                    'category': model_data.get('category', ''),
                    'description': model_data.get('description', '')
                }
                print_models([model_info], f"Model '{args.model_name}'")
            else:
                print(f"Model '{args.model_name}' not found.")
        return
    # Handle download use case
    if args.download:
        if not args.repo_id:
            print("Error: --repo-id is required for download")
            sys.exit(1)
        
        if not validate_repo_type(args.repo_type):
            sys.exit(1)
        
        # Track download timing and success
        start_time = datetime.now().isoformat()
        download_success = False
        
        try:
            # Get repository info and download
            metadata = get_repo_info(args.repo_id, args.token, args.repo_type, args.path, args.ngc_spec)
            if metadata is not None:
                download_repo(args.repo_id, args.token, args.path, args.repo_type, metadata, args.ngc_spec)
                download_success = True
            else:
                print("Error: Failed to get repository metadata")
        except Exception as e:
            print(f"Error during download: {str(e)}")
            download_success = False
        finally:
            finish_time = datetime.now().isoformat()
            # Update metadata file
            update_model_metadata(args.path, args.repo_type, args.repo_id, download_success, start_time, finish_time)
        
        if not download_success:
            sys.exit(1)
        sys.exit(0)
    

    if not upload_to_cloud(
        args.src, args.dst, args.cloud, args.token, args.recursive,
        args.endpoint, args.insecure, args.ca_bundle, args.account, args.container,
        args.repo_id, args.repo_type

    ):
        print(f"Error: Failed to upload repository: {args.repo_id} to {args.cloud}")
        sys.exit(1)
    print(f"Upload completed for repository: {args.repo_id} to {args.cloud}")

if __name__ == "__main__":
    main()